//
// Copyright 2019-2020 FRAGATA COMPUTER SYSTEMS AG
// Copyright 2017-2018 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//

//
// Based on neon, Intel(R) Nervana(tm) reference deep learning framework.
// Ported from Python to Go and partly modified by FRAGATA COMPUTER SYSTEMS AG.
//

package acc

import (
    "fmt"
    "fragata/arhat/backends"
    "fragata/arhat/base"
    "fragata/arhat/generators"
    "math"
)

//
//    Global device memory
//

type DeviceAllocation interface {
    Add(offset int) DeviceAllocation
}

//
//    AccTensor
//

type TakeArray struct {
    indices *AccTensor
    axis int
}

func NewTakeArray(indices *AccTensor, axis int) *TakeArray {
    return &TakeArray{indices, axis}
}

func(a *TakeArray) Indices() *AccTensor {
    return a.indices
}

func(a *TakeArray) Axis() int {
    return a.axis
}

type AccTensor struct {
    backends.TensorBase
    base *AccTensor
    size int
    strides []int
    nbytes int
    takeArray *TakeArray
    isTrans bool
    rounding int
    kahanCount int
    kahanReset int
    accdata DeviceAllocation
    isContiguous int
}

func NewAccTensor(
        backend AccGenerator,
        shape []int,
        dtype base.Dtype,
        name string,
        persistValues bool,
        tbase *AccTensor,
        accdata DeviceAllocation,
        strides []int,
        takeArray *TakeArray,
        isTrans bool,
        rounding int) *AccTensor {
    t := new(AccTensor)
    t.Init(
        t,
        backend,
        shape,
        dtype,
        name,
        persistValues,
        tbase,
        accdata,
        strides,
        takeArray,
        isTrans,
        rounding)
    return t
}

func(t *AccTensor) Init(
        self backends.Value,
        backend AccGenerator,
        shape []int,
        dtype base.Dtype,
        name string,
        persistValues bool,
        tbase *AccTensor,
        accdata DeviceAllocation,
        strides []int,
        takeArray *TakeArray,
        isTrans bool,
        rounding int) {
    dtype = base.ResolveDtype(dtype, base.Float32)
    rounding = base.ResolveInt(rounding, 0)

    // supported dtypes
    base.Assert(dtype == base.Float16 || dtype == base.Float32 ||
        dtype == base.Uint8 || dtype == base.Int8 ||
        dtype == base.Uint16 || dtype == base.Int16 ||
        dtype == base.Uint32 || dtype == base.Int32)

    shape = base.IntsExtend(shape, t.MinDims(), 1)

    t.TensorBase.Init(self, backend, shape, dtype, name, persistValues)

    size := base.IntsProd(shape)

    // only support C ordering for now
    if strides == nil {
        strides = ContiguousStrides(shape)
    }
    t.strides = strides

    t.base = tbase
    t.size = size
    t.nbytes = dtype.ItemSize() * size
    t.takeArray = takeArray
    t.isTrans = isTrans
    t.rounding = rounding
    t.kahanCount = 0
    t.kahanReset = 0

    if accdata == nil {
        if size != 0 {
            accdata = backend.MemAlloc(t.nbytes)
        }
        base.Assert(tbase == nil)
    }
    t.accdata = accdata

    t.isContiguous = base.IntNone // lazy evaluation
}

func(t *AccTensor) Base() backends.Tensor {
    return t.base
}

func(t *AccTensor) Size() int {
    return t.size
}

func(t *AccTensor) Strides() []int {
    return t.strides
}

func(t *AccTensor) TakeArray() *TakeArray {
    return t.takeArray
}

func(t *AccTensor) IsTrans() bool {
    return t.isTrans
}

func(t *AccTensor) Rounding() int {
    return t.rounding
}

func(t *AccTensor) AccData() DeviceAllocation {
    return t.accdata
}

func(t *AccTensor) Len() int {
    shape := t.Shape()
    if len(shape) != 0 {
        return shape[0]
    } else {
        return 0
    }
}

func(t *AccTensor) SetItem(index backends.Slice, value backends.Tensor) {
    t.GetItem(index).Assign(value)
}

func(t *AccTensor) GetItem(index backends.Slice) backends.Tensor {
    if index.IsNone() {
        // speed up common case of [:]
        return t
    }

    backend := t.Backend().(AccGenerator)
    dtype := t.Dtype()
    shape := t.Shape()
    strides := t.strides

    var newShape []int
    var newStrides []int
    newOffset := 0

    seenEllipsis := false
    var takeArray *TakeArray

    indexAxis := 0
    arrayAxis := 0

    fancyIndexing := func(x *AccTensor) {
        if x.Dtype() != base.Int32 {
            base.IndexError("Fancy indexing only currently supported with int32 types.")
        }

        if takeArray != nil {
            base.IndexError("Fancy indexing only currently supported one axis at a time.")
        }

        shape := x.Shape()
        size := 1
        for _, dim := range shape {
            if dim > size {
                if size != 1 {
                    base.IndexError(
                        "Fancy indexing only currently supported dim > 1 in a single dimension.")
                }
                size = dim
            }
        }

        takeArray = NewTakeArray(x, arrayAxis)
        
        newShape = append(newShape, size)
        newStrides = append(newStrides, strides[arrayAxis])
        
        indexAxis++
        arrayAxis++                
    }

    for indexAxis < index.Len() {
        indexEntry := index.Item(indexAxis)
        if arrayAxis > len(shape) {
            base.IndexError("too many axes in index")
        }

        switch v := indexEntry.(type) {
        case *backends.SliceItem:
            // Standard slicing (start:stop:step)
            start, stop, idxStrides := v.Indices(shape[arrayAxis])
            arrayStrides := strides[arrayAxis]
            newShape = append(newShape, (stop-start+idxStrides-1)/idxStrides)
            newStrides = append(newStrides, idxStrides*arrayStrides)
            newOffset += arrayStrides * start * dtype.ItemSize()
            indexAxis++
            arrayAxis++

        case *AccTensor:
            // fancy indexing: Tensor
            fancyIndexing(v)

        case int:
            arrayShape := shape[arrayAxis]
            if v < 0 {
                v += arrayShape
            }
            if v < 0 || v >= arrayShape {
                base.IndexError("subindex in axis %d out of range", indexAxis)
            }
            newOffset += strides[arrayAxis] * v * dtype.ItemSize()
            if len(shape) < 3 {
                newShape = append(newShape, 1)
                newStrides = append(newStrides, strides[arrayAxis])
            }
            indexAxis++
            arrayAxis++

        case backends.Ellipsis:
            indexAxis++
            remainingIndexCount := index.Len() - indexAxis
            newArrayAxis := len(shape) - remainingIndexCount
            if newArrayAxis < arrayAxis {
                base.IndexError("invalid use of ellipsis in index")
            }
            for arrayAxis < newArrayAxis {
                newShape = append(newShape, shape[arrayAxis])
                newStrides = append(newStrides, strides[arrayAxis])
                arrayAxis++
            }
            if seenEllipsis {
                base.IndexError("more than one ellipsis not allowed in index")
            }
            seenEllipsis = true

        default:
            base.IndexError("invalid subindex in axis %d", indexAxis)
        }
    }

    for arrayAxis < len(shape) {
        newShape = append(newShape, shape[arrayAxis])
        newStrides = append(newStrides, strides[arrayAxis])
        arrayAxis++
    }

    return NewAccTensor(
        backend,
        newShape,
        dtype,
        t.Name(),
        true,  // persistValues: default
        t,
        t.accdata.Add(newOffset),
        newStrides,
        takeArray,
        false, // isTrans: default
        t.rounding)
}

func(t *AccTensor) Assign(value backends.Value) backends.Tensor {
    backend := t.Backend().(AccGenerator)
    // ACHTUNG: CUDA streams are not supported in this release

    switch v := value.(type) {
    case *backends.Int:
        // if we have a contiguous array, then use the speedy driver kernel
        if t.IsContiguous() {
            x := v.Value()
            switch t.Dtype() {
            case base.Int8, base.Uint8:
                backend.MemsetD8Async(t.accdata, uint8(x), t.size)
            case base.Int16, base.Uint16:
                backend.MemsetD16Async(t.accdata, uint16(x), t.size)
            case base.Int32, base.Uint32:
                backend.MemsetD32Async(t.accdata, uint32(x), t.size)
            case base.Float16:
                half := base.NewHalf(float32(x))
                backend.MemsetD16Async(t.accdata, uint16(half), t.size)
            case base.Float32:
                backend.MemsetD32Async(t.accdata, math.Float32bits(float32(x)), t.size)
            default:
                base.NotImplementedError()
            }
        } else {
            // otherwise use our copy kernel
            backend.Assign(t, value)
        }

    case *backends.Float:
        // if we have a contiguous array, then use the speedy driver kernel
        if t.IsContiguous() {
            x := v.Value()
            switch t.Dtype() {
            case base.Int8, base.Uint8:
                backend.MemsetD8Async(t.accdata, uint8(x), t.size)
            case base.Int16, base.Uint16:
                backend.MemsetD16Async(t.accdata, uint16(x), t.size)
            case base.Int32, base.Uint32:
                backend.MemsetD32Async(t.accdata, uint32(x), t.size)
            case base.Float16:
                half := base.NewHalf(float32(x))
                backend.MemsetD16Async(t.accdata, uint16(half), t.size)
            case base.Float32:
                backend.MemsetD32Async(t.accdata, math.Float32bits(float32(x)), t.size)
            default:
                base.NotImplementedError()
            }
        } else {
            // otherwise use our copy kernel
            backend.Assign(t, value)
        }

    case *AccTensor:
        // TODO(orig): add an IsBinaryCompat like function
        if t.IsContiguous() && v.IsContiguous() && 
                t.Dtype() == v.Dtype() && base.IntsEq(t.Shape(), v.Shape()) {
            backend.MemcpyDtodAsync(t.accdata, v.accdata, t.nbytes)
        } else {
            backend.Assign(t, value)
        }

    case *backends.OpTreeNode:
        // collapse and execute an op tree as a kernel
        backend.Assign(t, value)

    default:
        base.TypeError("Invalid type for assignment: %t", value)
    }

    return t
}

func(t *AccTensor) GetScalar() backends.Value {
    backend := t.Backend().(AccGenerator)
    switch dtype := t.Dtype(); dtype {
    case base.Float16, base.Float32, base.Float64:
        v := backend.MakeFloatSymbol()
        s := v.Symbol()
        r := backend.GetFloat(t.accdata, dtype.ItemSize())
        backend.WriteLine("float %s = %s;", s, r)
        return v
    case base.Int8, base.Uint8, base.Int16, base.Uint16,
            base.Int32, base.Uint32, base.Int64, base.Uint64:
        v := backend.MakeIntSymbol()
        s := v.Symbol()
        r := backend.GetInt(t.accdata, dtype.ItemSize())
        backend.WriteLine("int %s = %s;", s, r)
        return v
    }
    base.Assert(false)
    return nil
}

func(t *AccTensor) Take(indices backends.Tensor, axis int) backends.Tensor {
    slice := backends.MakeSlice
    if axis == 1 {
        return t.GetItem(slice(nil, indices))
    } else {
        return t.GetItem(slice(indices, nil))
    }
}

func(t *AccTensor) Fill(value interface{}) backends.Tensor {
    // ACHTUNG: Add support for more types if needed
    switch v := value.(type) {
    case int:
        t.Assign(backends.NewInt(v))
    case float64:
        t.Assign(backends.NewFloat(v))
    default:
        base.NotImplementedError()
    }
    return t
}

func(t *AccTensor) Copy(x backends.Tensor) backends.Tensor {
    return t.Assign(x)
}

func(t *AccTensor) Reshape(shape []int) backends.Tensor {
    shape = base.IntsExtend(shape, t.MinDims(), 1)

    if base.IntsFind(shape, -1) >= 0 {
        // ACHTUNG: Multiple -1 case is apparently not allowed.
        //     This case will be likely detected by size check below.
        missingDim := -t.size / base.IntsProd(shape)
        newShape := make([]int, len(shape))
        for i, dim := range shape {
            if dim < 0 {
                dim = missingDim
            }
            newShape[i] = dim
        }
        shape = newShape
    }

    oldShape := t.Shape()
    if base.IntsEq(shape, oldShape) {
        return t
    }

    size := base.IntsProd(shape)

    if size != t.size {
        base.ValueError("total size of new array must be unchanged")
    }

    if t.takeArray != nil {
        base.ValueError("reshaping of non-contiguous arrays is not yet supported")
    }

    newStrides := reshapeStrides(t.strides, oldShape, shape)

    return NewAccTensor(
        t.Backend().(AccGenerator),
        shape,
        t.Dtype(),
        t.Name(),
        true,  // persistValues: default
        t,
        t.accdata,
        newStrides,
        nil,   // takeArray: default
        false, // isTrans: default
        t.rounding)
}

func(t *AccTensor) T() backends.Tensor {
    shape := t.Shape()
    strides := t.strides
    n := len(shape)
    switch n {
    case 1:
        shape = []int{shape[0]}
        strides = []int{strides[0]}
    case 2:
        shape = []int{shape[1], shape[0]}
        strides = []int{strides[1], strides[0]}
    default:
        // support for batched dot.
        // perserve outer dimension but reverse inner dims
        reverseInner := func(x []int) []int {
            n := len(x)
            y := make([]int, n)
            y[0] = x[0]
            for i := 1; i < n; i++ {
                y[i] = x[n-i]
            }
            return y
        }
        shape = reverseInner(shape)
        strides = reverseInner(strides)
    }

    return NewAccTensor(
        t.Backend().(AccGenerator),
        shape,
        t.Dtype(),
        t.Name(),
        true, // persistValues: default
        t.base,
        t.accdata,
        strides,
        nil,  // takeArray: default
        !t.isTrans,
        t.rounding)
}

func(t *AccTensor) Share(shape []int, dtype base.Dtype, name string) backends.Tensor {
    size := base.IntsProd(shape)
    if size > t.size {
        base.ValueError("total size of new array must <= size of parent")
    }

    if !t.IsContiguous() {
        base.TypeError("sharing of non-contigous arrays is not yet supported")
    }

    if dtype == base.DtypeNone {
        dtype = t.Dtype()
    }

    newBase := t.base
    if newBase == nil {
        newBase = t
    }

    return NewAccTensor(
        t.Backend().(AccGenerator),
        shape,
        dtype,
        name,
        true,  // persistValues: default
        newBase,
        t.accdata,
        ContiguousStrides(shape),
        nil,   // takeArray: default
        false, // isTrans: default
        t.rounding)
}

func(t *AccTensor) IsContiguous() bool {
    if t.isContiguous == base.IntNone {
        t.isContiguous = 0
        if t.takeArray == nil && base.IntsEq(t.strides, ContiguousStrides(t.Shape())) {
            t.isContiguous = 1
        }
    }
    return (t.isContiguous != 0)
}

//
//    AccGenerator
//

type AccGenerator interface {
    generators.Generator
    Assign(out *AccTensor, value backends.Value) backends.Value
    MemAlloc(nbytes int) DeviceAllocation
    MemsetD8Async(dest DeviceAllocation, data uint8, count int)
    MemsetD16Async(dest DeviceAllocation, data uint16, count int)
    MemsetD32Async(dest DeviceAllocation, data uint32, count int)
    MemcpyDtodAsync(dest DeviceAllocation, src DeviceAllocation, size int)
    GetInt(src DeviceAllocation, size int) string
    GetFloat(src DeviceAllocation, size int) string
}

//
//    AccGeneratorBase
//

//
//     ACHTUNG: Implementation of randState and hist* features is incomplete.
//     These features are not actually used in the current release.
//
//     randState is used to implement "rand" operation and random rounding;
//     it is indeed CUDA-specific and must be moved to "generators/cuda" package
//     or, preferrably, replaced with implementation based on CURAND device API.
//     
//     hist* will be used for implementation of callbacks in the future;
//     it is yet to be decided whether it shall stay here or moved to
//     platform-specific generators.
//

// ACHTUNG: Is RNG pool size backend-dependent?
//     (apparently yes; actually, it is CUDA-specific)

// size of the RNG pool on device
// currently this is hard wired
var rngPoolSize = []int{3 * 2048 * 32, 1}

type AccGeneratorBase struct {
    generators.GeneratorBase
    randState DeviceAllocation
    roundMode int
    buf map[string][]*AccTensor
    bufActive map[string][]*AccTensor
    histBins int
    histOffset int
    histMap map[string][]int // TODO: Revise this
    histIdx int
    histMax int
    histBase DeviceAllocation
}

func(b *AccGeneratorBase) Init(
        self generators.Generator,
        rngSeed int,
        defaultDtype base.Dtype,
        stochasticRound int,
        histBins int,
        histOffset int,
        compatMode backends.CompatMode) {
    defaultDtype = base.ResolveDtype(defaultDtype, base.Float32)
    stochasticRound = base.ResolveInt(stochasticRound, 0)
    histBins = base.ResolveInt(histBins, 64)
    histOffset = base.ResolveInt(histOffset, -48)

    if defaultDtype == base.Float32 {
        if stochasticRound != 0 {
            if stochasticRound == -1 {
                base.ValueError(
                    "Default rounding bit width is not supported for fp32. "+
                    "Please specify number of bits to round to.")
            }
/* TODO: Revise this (need logger)
            logger.Warn(
                "Using 32 bit floating point and setting stochastic rounding to %d bits", 
                    stochasticRound)
*/
        }
    }

    // super class init
    b.GeneratorBase.Init(self, rngSeed, defaultDtype, compatMode)

    // stochastic round
    // ACHTUNG: It is dififcult to interpret stochasticRound handling in original code.
    //     Foolowing is simplified variant.
    if stochasticRound == -1 {
        stochasticRound = 10
    }

    // attributes
    b.roundMode = stochasticRound

    b.buf = make(map[string][]*AccTensor)
    b.bufActive = make(map[string][]*AccTensor)

    // store histograms for batched memory
    b.histBins = base.IntNone
    b.histOffset = base.IntNone
    b.SetHistBuffers(histBins, histOffset)
}

// Backend methods

func(b *AccGeneratorBase) NewTensor(shape []int, dtype base.Dtype) backends.Tensor {
    return NewAccTensor(
        b.Self().(AccGenerator),
        shape,
        dtype,
        "",    // name
        true,  // persistValues
        nil,   // tbase
        nil,   // accdata
        nil,   // strides
        nil,   // takeArray
        false, // isTrans
        0)     // rounding
}

// helper methods

func(b *AccGeneratorBase) SetHistBuffers(histBins int, histOffset int) {
    if histBins != b.histBins || histOffset != b.histOffset {
        b.histBins = histBins
        b.histOffset = histOffset
        b.histMap = make(map[string][]int)
        b.histIdx = 0
        b.histMax = 4 * 4096
        self := b.Self().(AccGenerator)
        b.histBase = self.MemAlloc(b.histBins*b.histMax*4)
        // TODO: Must also generate code for filling histBase buffer
        //     with initial values. Add the respective code to prolog builder.
        // Use this pattern:
/*
        MemsetD32(b.histBase, 0, b.histBins*b.histMax)
*/
    }
}

func(b *AccGeneratorBase) GetRandStateDev() DeviceAllocation {
    if b.randState == nil {
        self := b.Self().(AccGenerator)
        size := base.IntsProd(rngPoolSize) * base.Uint32.ItemSize()
        b.randState = self.MemAlloc(size)
        // TODO: Must also generate code for filling randState buffer
        //     with initial values. Add the respective code to prolog builder
    }
    return b.randState
}

func(b *AccGeneratorBase) BufMalloc(shape []int) *AccTensor {
    var buf *AccTensor
    key := fmt.Sprintf("%v", shape)
    if list, ok := b.buf[key]; ok {
        if n := len(list); n != 0 {
            buf = list[n-1]
            b.buf[key] = list[:n-1]
        }
    }
    if buf == nil {
        self := b.Self().(AccGenerator)
        buf = self.NewTensor(shape, b.DefaultDtype()).(*AccTensor)
    }
    b.bufActive[key] = append(b.bufActive[key], buf)
    return buf
}

func(b *AccGeneratorBase) BufFree() {
    for key, src := range b.bufActive {
        dst := b.buf[key]
        for _, buf := range src {
            dst = append(dst, buf)
        }
        b.buf[key] = dst
        delete(b.bufActive, key)
    }
}

func(b *AccGeneratorBase) SplitToStacks(optree *backends.OpTreeNode) []backends.Stack {
    // post-order traversal
    wholeStack := optree.Traverse(nil)

    // build stages, each stage contains a sub optree
    var stages []*backends.OpTreeNode
    var mainStage []backends.Value
    var mainStageAxis []int

    stagesAppend := func(n *backends.OpTreeNode) {
        stages = append(stages, n)
    }
    mainStageAppend := func(v backends.Value) {
        mainStage = append(mainStage, v)
    }
    mainStageAxisAppend := func(axis int) {
        mainStageAxis = append(mainStageAxis, axis)
    }
    mainStagePop := func() backends.Value {
        n := len(mainStage) - 1
        v := mainStage[n]
        mainStage = mainStage[:n]
        return v
    }
    mainStageAxisPop := func() int {
        n := len(mainStageAxis) - 1
        v := mainStageAxis[n]
        mainStageAxis = mainStageAxis[:n]
        return v
    }

    // get minority axis for binary operation default, suports axis 0 and 1
    var axisCount [2]int
    for _, s := range wholeStack {
        if n, ok := s.(*backends.OpTreeNode); ok && backends.IsReductionOp(n.Op()) {
            axis := n.Axis()
            base.Assert(axis == 0 || axis == 1)
            axisCount[axis]++
        }
    }
    minorityAxis := 0
    if  axisCount[1] > axisCount[1] {
        minorityAxis = 1
    }

    node := backends.NewOpTreeNode
    assign := func(left backends.Value, right backends.Value) *backends.OpTreeNode {
        return backends.NewOpTreeNode(backends.Assign, left, right)
    }

    // traverse stack and split stages
    for _, s := range wholeStack {
        if n, ok := s.(*backends.OpTreeNode); ok {
            op := n.Op()
            switch {
            case op == backends.Dot:
                // convert left and right child to tensor when it was not
                right := mainStagePop()
                mainStageAxisPop() // don't care the value
                left := mainStagePop()
                mainStageAxisPop() // don't care the value
                if l, lok := left.(*backends.OpTreeNode); lok {
                    leftBuf := b.BufMalloc(l.Shape())
                    stagesAppend(assign(leftBuf, left))
                    left = leftBuf
                }
                if r, rok := right.(*backends.OpTreeNode); rok {
                    rightBuf := b.BufMalloc(r.Shape())
                    stagesAppend(assign(rightBuf, right))
                    right = rightBuf
                }
                // buffer to store the result of dot
                buf := b.BufMalloc([]int{left.Shape()[0], right.Shape()[1]})
                // save to stages
                stagesAppend(assign(buf, node(op, left, right)))
                // push buf to mainStage
                mainStageAppend(buf)
                mainStageAxisAppend(base.IntNone)

            case op == backends.Transpose:
                // the object being transposed must be optree here
                operand := mainStagePop()
                mainStageAxisPop() // don't care the value
                // allocate buf for the operand shape
                buf := b.BufMalloc(operand.Shape())
                // evaluate to buf
                stagesAppend(assign(buf, operand))
                // put the buf back to mainStage
                mainStageAppend(buf.T())
                mainStageAxisAppend(base.IntNone)

            case backends.IsReductionOp(op):
                // since 2d reduction is converted
                axis := n.Axis()
                base.Assert(axis != base.IntNone)
                operand := mainStagePop()
                prevAxis := mainStageAxisPop()
                if prevAxis != base.IntNone && prevAxis != axis {
                    // put everything under previous reduction to buf
                    buf := b.BufMalloc(operand.Shape())
                    stagesAppend(assign(buf, operand))
                    // put the buf with current reduction to main stage
                    mainStageAppend(node(op, buf, nil))
                    mainStageAxisAppend(axis)
                } else {
                    // do standard unary ops
                    mainStageAppend(node(op, operand, nil))
                    mainStageAxisAppend(axis)
                }

            case backends.IsUnaryOp(op):
                // will not run into multiple-axis reduction problem
                // just pop, build optree and put back
                operand := mainStagePop()
                axis := mainStageAxisPop()
                mainStageAppend(node(op, operand, nil))
                mainStageAxisAppend(axis) // cancelled out

            case backends.IsBinaryOp(op):
                // binary ops might run into multiple-axis reduction
                right := mainStagePop()
                prevAxisRight := mainStageAxisPop()
                left := mainStagePop()
                prevAxisLeft := mainStageAxisPop()
                if prevAxisRight != base.IntNone &&
                        prevAxisLeft != base.IntNone &&
                        prevAxisLeft != prevAxisRight {
                    // do reduction on minority axis
                    axis := base.IntNone
                    if prevAxisLeft == minorityAxis {
                        buf := b.BufMalloc(left.Shape())
                        stagesAppend(assign(buf, left))
                        left = buf
                        axis = prevAxisRight
                    } else {
                        buf := b.BufMalloc(right.Shape())
                        stagesAppend(assign(buf, right))
                        right = buf
                        axis = prevAxisLeft
                    }
                    // append to main stage
                    mainStageAppend(node(op, left, right))
                    mainStageAxisAppend(axis)
                } else {
                    // no multiple-axis reduction, perform standard process
                    mainStageAppend(node(op, left, right))
                    axis := base.IntNone
                    if prevAxisLeft != base.IntNone {
                        axis = prevAxisLeft
                    } else {
                        axis = prevAxisRight
                    }
                    mainStageAxisAppend(axis)
                }

            default:
                base.NotImplementedError()
            }
        } else {
            // tensor or scalars, just push to mainStage
            mainStageAppend(s)
            mainStageAxisAppend(base.IntNone)
        }
    }

    // append to the last stage
    stagesAppend(mainStage[0].(*backends.OpTreeNode))

    // build stacks for CallCompoundKernel
    var stacks []backends.Stack
    for _, stage := range stages {
        // now all stages is exect one simple optree
        // create stack
        stacks = append(stacks, stage.Traverse(nil))
    }

    // free buffer from bufActive to buf, without loosing the reference
    b.BufFree()

    return stacks
}

func IsSimpleStack(stack backends.Stack) bool {
    var reductionAxes [2]bool
    for _, s := range stack {
        if n, ok := s.(*backends.OpTreeNode); ok {
            op := n.Op()
            if op == backends.Dot || op == backends.Transpose {
                return false
            }
            if backends.IsReductionOp(op) {
                axis := n.Axis()
                reductionAxes[axis] = true
                if reductionAxes[1-axis] {
                    return false
                }
            }
        }
    }
    return true
}

//
//    Public helper functions
//

func ContiguousStrides(shape []int) []int {
    n := len(shape)
    if n == 0 {
        return nil
    }
    strides := make([]int, n)
    strides[n-1] = 1
    for i := n - 1; i > 0; i-- {
        strides[i-1] = strides[i] * shape[i]
    }
    return strides
}

//
//    Local helper functions
//

func reshapeStrides(origStrides []int, origShape []int, newShape[] int) []int {
    // Only contiguous dimensions can be reshaped
    matchedDims := 0
    newLen := len(newShape)
    n := base.IntMin(len(origShape), newLen)
    for i := 0; i < n; i++ {
        if origShape[i] != newShape[i] {
            break;
        }
        matchedDims++
    }

    // ACHTUNG: Why do they need this in original code?
    //     It apparently doesn't affect the following operations
    // Extend original shape to length of new shape
    origShape = base.IntsExtend(origShape, newLen, 1)
    origStrides = base.IntsExtend(origStrides, newLen, 1)

    reshapeSize := base.IntsProd(newShape[matchedDims:])
    origSize := origStrides[matchedDims] * origShape[matchedDims]
    if origSize != reshapeSize {
        base.ValueError("Reshaping of non-contiguous dimensions unsupported.")
    }

    newStrides := make([]int, newLen)
    copy(newStrides[:matchedDims], origStrides)
    copy(newStrides[matchedDims:], ContiguousStrides(newShape[matchedDims:]))
    return newStrides
}

