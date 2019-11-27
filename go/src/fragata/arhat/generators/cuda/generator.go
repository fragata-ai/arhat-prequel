//
// Copyright 2019 FRAGATA COMPUTER SYSTEMS AG
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

package cuda

import (
    "fmt"
    "fragata/arhat/backends"
    "fragata/arhat/base"
    "fragata/arhat/generators"
    "fragata/arhat/generators/acc"
    kernels "fragata/arhat/kernels/cuda"
    "math"
    "os"
    "path"
    "strings"
)

//
//    Global device memory
//

type CudaDeviceAllocation struct {
    index int
    nbytes int
    offset int
}

func NewCudaDeviceAllocation(index int, nbytes int, offset int) *CudaDeviceAllocation {
    return &CudaDeviceAllocation{index, nbytes, offset}
}

func(a *CudaDeviceAllocation) Add(offset int) acc.DeviceAllocation {
    nbytes := a.nbytes - offset
    base.Assert(nbytes >= 0)
    return NewCudaDeviceAllocation(a.index, nbytes, a.offset+offset)
}

//
//    CudaGenerator
//

type CudaGenerator struct {
    acc.AccGeneratorBase
    deviceType int
    deviceId int
    scratchData acc.DeviceAllocation
    scratchSize int
    scratchOffset int
    bench bool
    computeCapability [2]int
    smCount int
    usePinnedMem bool
    buffers []*CudaDeviceAllocation
    nextBufferIndex int
    nextKernelIndex int
    buildMainCpp bool
    filePrefix string
    hostNamespace string
    kernelPrefix string
    kernelCache map[string]*Kernel
    kernelIdMap map[*Kernel]string
}

func NewCudaGenerator(
        rngSeed int,
        defaultDtype base.Dtype,
        stochasticRound int,
        deviceId int,
        computeCapability [2]int,
        bench bool,
        scratchSize int,
        histBins int,
        histOffset int,
        compatMode backends.CompatMode) *CudaGenerator {
    b := new(CudaGenerator)
    b.Init(
        b,
        rngSeed,
        defaultDtype,
        stochasticRound,
        deviceId,
        computeCapability,
        bench,
        scratchSize,
        histBins,
        histOffset,
        compatMode)
    return b
}

func(b *CudaGenerator) Init(
        self generators.Generator,
        rngSeed int,
        defaultDtype base.Dtype,
        stochasticRound int,
        deviceId int,
        computeCapability [2]int,
        bench bool,
        scratchSize int,
        histBins int,
        histOffset int,
        compatMode backends.CompatMode) {
    defaultDtype = base.ResolveDtype(defaultDtype, base.Float32)
    stochasticRound = base.ResolveInt(stochasticRound, 0)
    deviceId = base.ResolveInt(deviceId, 0)
    scratchSize = base.ResolveInt(scratchSize, 0)
    histBins = base.ResolveInt(histBins, 64)
    histOffset = base.ResolveInt(histOffset, -48)

    if defaultDtype != base.Float16 && defaultDtype != base.Float32 {
        base.ValueError("Default data type for cuda backend must be float16 or 32")
    }

    // AccGeneratorBase.Init allocates device memory buffers;
    //     therefore buffer index must be initialized before
    b.nextBufferIndex = 0

    b.AccGeneratorBase.Init(
        self,
        rngSeed,
        defaultDtype,
        stochasticRound,
        histBins,
        histOffset,
        compatMode)

    b.deviceType = 1
    b.deviceId = deviceId

    // attributes
    b.scratchData = nil
    b.scratchSize = scratchSize
    b.scratchOffset = 0
    b.bench = bench

    b.computeCapability = computeCapability
    b.smCount = base.IntNone // optionally set via SetSmCount

    b.usePinnedMem = true

    b.nextKernelIndex = 1

    b.buildMainCpp = true

    b.kernelCache = make(map[string]*Kernel)
    b.kernelIdMap = make(map[*Kernel]string)
}

func(b *CudaGenerator) ConfigureCodeOutput(
        buildMainCpp bool,
        filePrefix string, 
        hostNamespace string, 
        kernelPrefix string) {
    b.buildMainCpp = buildMainCpp
    b.filePrefix = filePrefix
    b.hostNamespace = hostNamespace
    b.kernelPrefix = kernelPrefix
}

func(b *CudaGenerator) SetSmCount(smCount int) {
    b.smCount = smCount
}

func(b *CudaGenerator) GetSmCount() int {
    return b.smCount
}

func(b *CudaGenerator) ScratchBufferInit() {
    b.scratchOffset = 0
}

func(b *CudaGenerator) ScratchBufferReset() {
    b.scratchData = nil
    b.scratchSize = 0
    b.scratchOffset = 0
}

func(b *CudaGenerator) ScratchBufferOffset(size int) acc.DeviceAllocation {
    if size & 127 != 0 {
        size += 128 - (size & 127)
    }

    if size + b.scratchOffset > b.scratchSize {
        base.RuntimeError(
            "CudaGenerator.scratchSize(%d) is too small for this operation(%d, %d)",
                b.scratchSize, size, b.scratchOffset)
    }

    if b.scratchData == nil {
        b.scratchData = b.MemAlloc(b.scratchSize)
    }

    data := b.scratchData.Add(b.scratchOffset)
    b.scratchOffset += size

    return data
}

func(b *CudaGenerator) SetScratchSize(args ...int) {
    totalSize := 0
    for _, size := range args {
        if size & 127 != 0 {
            size += 128 - (size & 127)
        }
        totalSize += size
    }

    if totalSize > b.scratchSize {
        b.scratchSize = totalSize
    }
}

func(b *CudaGenerator) LookupKernel(key string) *Kernel {
    kernel, ok := b.kernelCache[key]
    if !ok {
        return nil
    }
    // debugging hooks can be placed here
    return kernel
}

func(b *CudaGenerator) RegisterKernel(key string, kernel *Kernel) {
    b.kernelCache[key] = kernel
}

// Backends methods

func(b *CudaGenerator) RngNormal(out backends.Tensor, loc float64, scale float64, size []int) {
    b.WriteLine("RngNormal((float *)%s, %s, %s, %d);",
        formatBufferRef(out.(*acc.AccTensor).AccData().(*CudaDeviceAllocation), true),
        formatFloat32(loc),
        formatFloat32(scale),
        base.IntsProd(size))
}

func(b *CudaGenerator) RngUniform(out backends.Tensor, low float64, high float64, size []int) {
    b.WriteLine("RngUniform((float *)%s, %s, %s, %d);",
        formatBufferRef(out.(*acc.AccTensor).AccData().(*CudaDeviceAllocation), true),
        formatFloat32(low),
        formatFloat32(high),
        base.IntsProd(size))
}

func(b *CudaGenerator) CompoundDot(
        x backends.Tensor,
        y backends.Tensor,
        z backends.Tensor,
        alpha float64,
        beta float64,
        relu bool,
        bsum backends.Tensor) backends.Tensor {
    cudaBsum, _ := bsum.(*acc.AccTensor) // may be nil
    return b.CompoundDotCuda(
        x.(*acc.AccTensor),
        y.(*acc.AccTensor),
        z.(*acc.AccTensor),
        alpha,
        beta,
        relu,
        cudaBsum,
        1, 
        nil)
}

func(b *CudaGenerator) CompoundDotCuda(
        x *acc.AccTensor,
        y *acc.AccTensor,
        z *acc.AccTensor,
        alpha float64, 
        beta float64,
        relu bool,
        bsum *acc.AccTensor,
        repeat int,
        size []int) *acc.AccTensor {
    // ACHTUNG: Only CudaC implementation is currently supported
    // SKIPPED: Support for 'repeat'
    b.cublasDot(x, y, z, alpha, beta)
    // ACHTUNG: relu not supported in original code; modelled here after CPU backend
    if relu {
        z.Assign(b.Maximum(z, b.Float(0.0)))
    }
    if bsum != nil {
        bsum.Assign(b.Sum(z, 1))
    }
    return z
}

func(b *CudaGenerator) cublasDot(
        x *acc.AccTensor,
        y *acc.AccTensor,
        z *acc.AccTensor,
        alpha float64, 
        beta float64) {
    ldx := base.IntsMax(x.Strides())
    ldy := base.IntsMax(y.Strides())
    ldz := base.IntsMax(z.Strides())

    opX := 'n'
    if x.IsTrans() {
        opX = 't'
    }
    opY := 'n'
    if y.IsTrans() {
        opY = 't'
    }

    xshape := x.Shape()
    yshape := y.Shape()
    zshape := z.Shape()

    m := xshape[0]
    n := yshape[1]
    k := xshape[1]

    xDtype := x.Dtype()
    canFloat16 := 
        ((b.computeCapability[0] == 5 && b.computeCapability[1] >= 2) ||
            b.computeCapability[0] >= 6)
    falpha := formatFloat32(alpha)
    fbeta := formatFloat32(beta)
    fmtData := func(a acc.DeviceAllocation) string {
        return formatBufferRef(a.(*CudaDeviceAllocation), true)
    }

    // Swap X and Y to map from C order to Fortran
    switch {
    case xDtype == base.Float32 || (xDtype == base.Float16 && canFloat16):
        // TODO: This case requires further revision as in C/C++ separate set of
        //     functions will be apparently needed to support float16 (CublasHgem*)
        xdata := fmtData(x.AccData())
        ydata := fmtData(y.AccData())
        zdata := fmtData(z.AccData())

        if n != 1 || (opX == 't' && opY == 'n') {
            b.writeLongStmt(
                "CublasSgemm('%c', '%c', %d, %d, %d,#"+
                    "%s, (float *)%s, %d, (float *)%s, %d,#"+
                    "%s, (float *)%s, %d);",
                        opY, opX, n, m, k, falpha, ydata, ldy, xdata, ldx, fbeta, zdata, ldz)
        } else {
            b.writeLongStmt(
                "CublasSgemv('t', %d, %d,#"+
                    "%s, (float *)%s, %d, (float *)%s, %d,#"+
                    "%s, (float *)%s, %d);",
                        k, m, falpha, xdata, k, ydata, ldy, fbeta, zdata, ldz)
        }

    case xDtype == base.Float16:
        // fp16 gemm not supported by cublas until 7.5, so do conversion
        xtemp := b.BufMalloc([]int{xshape[0], xshape[1]*2})
        ytemp := b.BufMalloc([]int{yshape[0], yshape[1]*2})
        ztemp := b.BufMalloc([]int{zshape[0], zshape[1]*2})

        xfp32 := 
            acc.NewAccTensor(
                b, 
                xshape, 
                base.Float32, 
                "",              // name
                true,            // persistValues
                nil,             // tbase
                xtemp.AccData(),
                x.Strides(),
                nil,             // takeArray
                x.IsTrans(),
                base.IntNone)    // rounding
        yfp32 := 
            acc.NewAccTensor(
                b, 
                yshape, 
                base.Float32, 
                "",              // name
                true,            // persistValues
                nil,             // tbase
                ytemp.AccData(),
                y.Strides(),
                nil,             // takeArray
                y.IsTrans(),
                base.IntNone)    // rounding
        zfp32 := 
            acc.NewAccTensor(
                b, 
                zshape, 
                base.Float32, 
                "",              // name
                true,            // persistValues
                nil,             // tbase
                ztemp.AccData(),
                z.Strides(),
                nil,             // takeArray
                z.IsTrans(),
                base.IntNone)    // rounding

        xfp32.Assign(x)
        yfp32.Assign(y)
        zfp32.Assign(z)

        xdata := fmtData(xfp32.AccData())
        ydata := fmtData(yfp32.AccData())
        zdata := fmtData(zfp32.AccData())

        if n != 1 || (opX == 't' && opY == 'n') {
            b.writeLongStmt(
                "CublasSgemm('%c', '%c', %d, %d, %d,#"+
                    "%s, (float *)%s, %d, (float *)%s, %d,#"+
                    "%s, (float *)%s, %d);",
                        opY, opX, n, m, k, falpha, ydata, ldy, xdata, ldx, fbeta, zdata, ldz)
        } else {
            b.writeLongStmt(
                "CublasSgemv('t', %d, %d,#"+
                    "%s, (float *)%s, %d, (float *)%s, %d,#"+
                    "%s, (float *)%s, %d);",
                        k, m, falpha, xdata, k, ydata, ldy, fbeta, zdata, ldz)
        }

        z.Assign(zfp32)
        b.BufFree()

    default:
        base.TypeError("Unsupported type for cublas gemm")
    }
}

func(b *CudaGenerator) MakeBinaryMask(out backends.Tensor, keepThresh float64) {
    // generate random number uniformly distributed between 0 and 1
    b.RngUniform(out, 0.0, 1.0, out.Shape())
    out.Assign(b.LessEqual(out, b.Float(keepThresh)))
}

func(b *CudaGenerator) Binarize(
        ary backends.Tensor, out backends.Tensor, stochastic bool) backends.Tensor {
    // TODO
    base.NotImplementedError()
    return nil
}

func(b *CudaGenerator) BuildConvKernels(dtype base.Dtype, a *ConvParams) (
        ConvFpropKernels, ConvBpropKernels, ConvUpdateKernels) {
    // implements interface ConvKernelBuilder
    fpropKernels := NewFpropCuda(b, dtype, a)
    bpropKernels := NewBpropCuda(b, dtype, a)
    updateKernels := NewUpdateCuda(b, dtype, a)
    return fpropKernels, bpropKernels, updateKernels
}

func(b *CudaGenerator) NewConvLayer(
        dtype base.Dtype, params *backends.ConvParams) backends.ConvLayer {
    return NewConvLayer(b, dtype, params, b)
}

func(b *CudaGenerator) NewDeconvLayer(
        dtype base.Dtype, params *backends.DeconvParams) backends.DeconvLayer {
    return NewDeconvLayer(b, dtype, params, b)
}

// SKIPPED: repeat (gpu only), layerOp
func(b *CudaGenerator) FpropConv(
        layer backends.ConvLayerBase,
        i backends.Tensor,
        f backends.Tensor,
        o backends.Tensor,
        x backends.Tensor,
        bias backends.Tensor,
        bsum backends.Tensor,
        alpha float64,
        beta float64,
        relu bool,
        brelu bool,
        slope float64) {
    base.Assert(layer.SizeI() == i.Size())
    base.Assert(layer.SizeF() == f.Size())
    base.Assert(layer.SizeO() == o.Size())

    cudaLayer := getCudaConvLayer(layer)
    fpropKernels := cudaLayer.fpropKernels
    fpropKernels.BindParams(i, f, o, x, bias, bsum, alpha, beta, relu, brelu, slope)
    b.executeFpropConv(cudaLayer, fpropKernels)
    // SKIPPED: Returned value
}

// SKIPPED: repeat (gpu only), layerOp
func(b *CudaGenerator) BpropConv(
        layer backends.ConvLayerBase,
        f backends.Tensor,
        e backends.Tensor,
        gradI backends.Tensor,
        x backends.Tensor,
        bias backends.Tensor,
        bsum backends.Tensor,
        alpha float64,
        beta float64,
        relu bool,
        brelu bool,
        slope float64) {
    base.Assert(layer.SizeF() == f.Size())
    base.Assert(layer.SizeO() == e.Size())
    base.Assert(layer.SizeI() == gradI.Size())

    cudaLayer := getCudaConvLayer(layer)
    bpropKernels := cudaLayer.bpropKernels
    bpropKernels.BindParams(e, f, gradI, x, bias, bsum, alpha, beta, relu, brelu, slope)
    b.executeBpropConv(cudaLayer, bpropKernels)
    // SKIPPED: Returned value
}

// SKIPPED: repeat (gpu only), layerOp
func(b *CudaGenerator) UpdateConv(
        layer backends.ConvLayerBase,
        i backends.Tensor,
        e backends.Tensor,
        gradF backends.Tensor,
        alpha float64,
        beta float64,
        gradBias backends.Tensor) {
    base.Assert(layer.SizeI() == i.Size())
    base.Assert(layer.SizeO() == e.Size())
    base.Assert(layer.SizeF() == gradF.Size())

    cudaLayer := getCudaConvLayer(layer)
    if cudaLayer.nck[0] < 4 && base.IntsEq(cudaLayer.trs, []int{1, 1, 1}) {
        ir := i.Reshape([]int{cudaLayer.nck[1], -1})
        er := e.Reshape([]int{cudaLayer.nck[2], -1})
        gr := gradF.Reshape([]int{cudaLayer.nck[1], -1})
        b.CompoundDot(ir, er, gr, alpha, beta, false, nil)
    } else {
        cudaLayer.updateKernels.BindParams(i, e, gradF, alpha, beta, false)
        b.executeUpdateConv(cudaLayer, cudaLayer.updateKernels)
    }
    // SKIPPED: Returned value
}

// SKIPPED: Support for repeat
func(b *CudaGenerator) executeFpropConv(layer *ConvLayerBase, kernels ConvFpropKernels) {
    // SKIPPED: Support for repeat and bench
    kernels.Execute()
    // SKIPPED: Returned value
}

// SKIPPED: Support for repeat
func(b *CudaGenerator) executeBpropConv(layer *ConvLayerBase, kernels ConvBpropKernels) {
    // SKIPPED: Support for repeat and bench
    kernels.Execute()
    // SKIPPED: Returned value
}

// SKIPPED: Support for repeat
func(b *CudaGenerator) executeUpdateConv(layer *ConvLayerBase, kernels ConvUpdateKernels) {
    // SKIPPED: Support for repeat and bench
    kernels.Execute()
    // SKIPPED: Returned value
}

func getCudaConvLayer(layer backends.ConvLayerBase) *ConvLayerBase {
    switch v := layer.(type) {
    case *ConvLayer:
        return &v.ConvLayerBase
    case *DeconvLayer:
        return &v.ConvLayerBase
    default:
        base.AssertionError("Invalid layer type")
        return nil
    }
}

func(b *CudaGenerator) NewLrnLayer(
        dtype base.Dtype, params *backends.LrnParams) backends.LrnLayer {
    return NewLrnLayer(b, dtype, params)
}

// SKIPPED: Support for 'repeat'
func(b *CudaGenerator) FpropLrn(
        layer backends.LrnLayer, 
        i backends.Tensor,
        o backends.Tensor,
        denom backends.Tensor,
        alpha float64,
        beta float64,
        ascale float64,
        bpower float64) {
    base.Assert(layer.SizeI() == i.Size())
    base.Assert(layer.SizeO() == o.Size())
    cudaLayer := layer.(*LrnLayer)
    kernelSpec := cudaLayer.fpropKernel
    shared := cudaLayer.bpropLutSize // ACHTUNG: Why not fpropLutSize?
    b.executeLrn(
        false,
        cudaLayer,
        i,
        o,
        nil,
        nil,
        denom,
        alpha,
        beta,
        ascale,
        bpower,
        kernelSpec,
        shared)
}

// SKIPPED: Support for 'repeat'
func(b *CudaGenerator) BpropLrn(
        layer backends.LrnLayer, 
        i backends.Tensor,
        o backends.Tensor,
        e backends.Tensor,
        delta backends.Tensor,
        denom backends.Tensor,
        alpha float64,
        beta float64,
        ascale float64,
        bpower float64) {
    base.Assert(layer.SizeI() == i.Size())
    base.Assert(layer.SizeO() == e.Size())
    base.Assert(layer.SizeI() == delta.Size())
    cudaLayer := layer.(*LrnLayer)
    kernelSpec := cudaLayer.bpropKernel
    shared := cudaLayer.bpropLutSize
    b.executeLrn(
        true,
        cudaLayer,
        i,
        o,
        e,
        delta,
        denom,
        alpha,
        beta,
        ascale,
        bpower,
        kernelSpec,
        shared)
}

// SKIPPED: Support for 'repeat'
func(b *CudaGenerator) executeLrn(
        backprop bool,
        layer *LrnLayer,
        i backends.Tensor,
        o backends.Tensor,
        e backends.Tensor,
        delta backends.Tensor,
        denom backends.Tensor,
        alpha float64,
        beta float64,
        ascale float64,
        bpower float64,
        kernelSpec *PoolKernel,
        shared int) {
    base.Assert(i.Dtype() == o.Dtype())
    kernel := b.MapStringToFunc(kernelSpec.name, layer.dtype)
    flags := 0
    var params []acc.KernelArgument
    // SKIPPED: Stream as params[0]
    if !backprop {
        params = []acc.KernelArgument{i, o, denom, alpha, beta, ascale, bpower, flags}
    } else {
        params = []acc.KernelArgument{i, o, e, delta, denom, alpha, beta, ascale, bpower, flags}
    }
    // SKIPPED: Support for repeat and bench
    kernel.Launch(kernelSpec.grid, kernelSpec.block, shared, params, kernelSpec.args)
}

func(b *CudaGenerator) NewPoolLayer(
        dtype base.Dtype, params *backends.PoolParams) backends.PoolLayer {
    return NewPoolLayer(b, dtype, params)
}

// SKIPPED: Support for 'repeat'
func(b *CudaGenerator) FpropPool(
        layer backends.PoolLayer, 
        i backends.Tensor, 
        o backends.Tensor, 
        argmax backends.Tensor, 
        alpha float64, 
        beta float64) {
    base.Assert(layer.SizeI() == i.Size())
    base.Assert(layer.SizeO() == o.Size())
    if layer.Op() == backends.PoolOpMax {
        base.AssertMsg(argmax != nil, "max pooling requires argmax buffer")
    }
    cudaLayer := layer.(*PoolLayer)
    b.executePool(
        cudaLayer,
        i,
        o,
        argmax,
        alpha,
        beta,
        cudaLayer.fpropKernel,
        cudaLayer.fpropLutSize)
    // SKIPPED: Returned value
}

// SKIPPED: repeat (gpu only)
func(b *CudaGenerator) BpropPool(
        layer backends.PoolLayer, 
        i backends.Tensor, 
        o backends.Tensor, 
        argmax backends.Tensor, 
        alpha float64, 
        beta float64) {
    base.AssertMsg(layer.SizeI() == o.Size(),
        "mismatch between sizeI %d and O %d", layer.SizeI(), o.Size())
    base.AssertMsg(layer.SizeO() == i.Size(),
        "mismatch between sizeO %d and I %d", layer.SizeO(), i.Size())
    if layer.Op() == backends.PoolOpMax {
        base.AssertMsg(argmax != nil, "max pooling requires argmax buffer")
    }
    if argmax != nil {
        base.AssertMsg(layer.SizeO() == argmax.Size(), 
            "Pooling argmax size does not match input size!")
    }
    base.Assert(i.Dtype() == o.Dtype())
    cudaLayer := layer.(*PoolLayer)
    b.executePool(
        cudaLayer,
        i,
        o,
        argmax,
        alpha,
        beta,
        cudaLayer.bpropKernel,
        cudaLayer.bpropLutSize)
    // SKIPPED: Returned value
}

// SKIPPED: repeat (gpu only)
func(b *CudaGenerator) executePool(
        layer *PoolLayer,
        i backends.Tensor,
        o backends.Tensor,
        argmax backends.Tensor,
        alpha float64,
        beta float64,
        kernelSpec *PoolKernel,
        shared int) {
    // SKIPPED: Support for repeat and bench
    base.Assert(i.Dtype() == o.Dtype())
    kernel := b.MapStringToFunc(kernelSpec.name, layer.dtype)
    flags := 0
    // SKIPPED: Stream as params[0]
    params := []acc.KernelArgument{i, o, argmax, alpha, beta, flags}
    // SKIPPED: Support for repeat and bench
    kernel.Launch(kernelSpec.grid, kernelSpec.block, shared, params, kernelSpec.args)
}

func(b *CudaGenerator) MapStringToFunc(funcname string, dtype base.Dtype) *Kernel {
    key := 
        fmt.Sprintf("%s_%s_%d_%d", 
            funcname, dtype, b.computeCapability[0], b.computeCapability[1])
    kernel := b.LookupKernel(key)
    if kernel != nil {
        return kernel
    }
    name, code := kernels.MapStringToFunc(funcname, dtype, b.computeCapability)
    kernel = NewKernel(b, name, code)
    b.RegisterKernel(key, kernel)
    return kernel
}

func(b *CudaGenerator) NewReluLayer() backends.Layer {
    // not used with this backend
    return nil
}

func(b *CudaGenerator) FpropRelu(
        layer backends.Layer,
        x backends.Tensor, 
        slope float64) backends.Value {
   // maximum(x, 0) + slope * minimum(0, x)
    zero := b.Float(0.0)
    return b.Maximum(x, zero).Add(b.Float(slope).Mul(b.Minimum(x, zero)))
}

func(b *CudaGenerator) BpropRelu(
        layer backends.Layer,
        x backends.Tensor,
        errors backends.Tensor,
        deltas backends.Tensor,
        slope float64) backends.Value {
    // greater(x, 0) + slope * less(x, 0)
    zero := b.Float(0.0)
    return b.Greater(x, zero).Add(b.Float(slope).Mul(b.Less(x, zero)))
}

func(b *CudaGenerator) NewBatchNormLayer(inShape []int) backends.BatchNormLayer {
    // not used with this backend
    return nil
}

// SKIPPED: threads, repeat (both gpu only)
func(b *CudaGenerator) CompoundFpropBn(
        x backends.Tensor,
        xsum backends.Tensor,
        xvar backends.Tensor,
        gmean backends.Tensor,
        gvar backends.Tensor,
        gamma backends.Tensor,
        beta backends.Tensor,
        y backends.Tensor,
        eps float64,
        rho float64,
        computeBatchSum bool,
        accumbeta float64,
        relu bool,
        binary bool,
        inference bool,
        outputs backends.Tensor,
        layer backends.BatchNormLayer) {
    base.Assert(xsum.Dtype() == base.Float32)

    if inference {
        // xhat = (x - gmean) / sqrt(gvar + eps)
        // y[] = y * accumbeta + xhat * gamma + beta
        xhat := x.Sub(gmean).Div(b.Sqrt(gvar.Add(b.Float(eps)))) // Op-tree only
        y.Assign(y.Mul(b.Float(accumbeta)).Add(xhat.Mul(gamma)).Add(beta))
        return
    }

    if computeBatchSum {
        xsum.Assign(b.Sum(x, 1))
    }

    xshape := x.Shape()
    k := xshape[0]
    n := xshape[1]

    // SKIPPED: threads passed by callse
    threads := base.IntNone
    if n <= 8192 {
        threads = 1 << uint(base.IntMax(5, int(math.Round(math.Log2(float64(n))))-3))
    } else {
        smCount := b.GetSmCount()
        if smCount != base.IntNone {
            occup := k / (128.0 * smCount)
            for t := 32; t <= 512; t *= 2 {
                if occup * t > 5.0 {
                    threads = t
                    break
                }
            }
        }
        if threads == base.IntNone {
            threads = 1024
        }
    }

    grid := []int{k, 1, 1}
    block := []int{threads, 1, 1}
    // SKIPPED: stream as params[0]
    params := []acc.KernelArgument{
        y, 
        xvar, 
        gmean, 
        gvar, 
        x, 
        xsum, 
        gmean, 
        gvar, 
        gamma, 
        beta, 
        eps, 
        rho, 
        accumbeta, 
        n, 
        relu, 
        binary,
    }

    kernel := b.getBnFpropKernel(x.Dtype(), threads)

    b.executeBn(kernel, grid, block, params)
}

func(b *CudaGenerator) getBnFpropKernel(dtype base.Dtype, threads int) *Kernel {
    key := 
        fmt.Sprintf("bn_fprop_%s_%d_%d_%d",
            dtype, threads, b.computeCapability[0], b.computeCapability[1])
    kernel := b.LookupKernel(key)
    if kernel != nil {
        return kernel
    }
    name, code := kernels.GetBnFpropKernel(dtype, threads, b.computeCapability)
    kernel = NewKernel(b, name, code)
    b.RegisterKernel(key, kernel)
    return kernel
}

// SKIPPED: threads, repeat (both gpu only)
func(b *CudaGenerator) CompoundBpropBn(
        deltaOut backends.Tensor,
        gradGamma backends.Tensor,
        gradBeta backends.Tensor,
        deltaIn backends.Tensor,
        x backends.Tensor,
        xsum backends.Tensor,
        xvar backends.Tensor,
        gamma backends.Tensor,
        eps float64,
        binary bool,
        layer backends.BatchNormLayer) {
    base.AssertMsg(xsum.Dtype() == base.Float32, "xsum should be fp32")

    xshape := x.Shape()
    k := xshape[0]
    n := xshape[1]

    // SKIPPED: threads passed by callse
    threads := base.IntNone
    if n <= 8192 {
        threads = 1 << uint(base.IntMax(5, int(math.Round(math.Log2(float64(n))))-3))
    } else {
        if k < 192 {
            threads = 128
        } else {
            threads = 64
        }
    }

    grid := []int{k, 1, 1}
    block := []int{threads, 1, 1}
    // SKIPPED: stream as params[0]
    params := []acc.KernelArgument{
        deltaOut, 
        gradGamma, 
        gradBeta, 
        deltaIn,
        x, 
        xsum, 
        xvar, 
        gamma, 
        eps, 
        n, 
        binary,
    }

    kernel := b.getBnBpropKernel(x.Dtype(), threads)

    b.executeBn(kernel, grid, block, params)
}

func(b *CudaGenerator) getBnBpropKernel(dtype base.Dtype, threads int) *Kernel {
    key := 
        fmt.Sprintf("bn_bprop_%s_%d_%d_%d",
            dtype, threads, b.computeCapability[0], b.computeCapability[1])
    kernel := b.LookupKernel(key)
    if kernel != nil {
        return kernel
    }
    name, code := kernels.GetBnBpropKernel(dtype, threads, b.computeCapability)
    kernel = NewKernel(b, name, code)
    b.RegisterKernel(key, kernel)
    return kernel
}

// SKIPPED: repeat, size, n
func(b *CudaGenerator) executeBn(
        kernel *Kernel,
        grid []int,
        block []int,
        params []acc.KernelArgument) {
    // SKIPPED: Support for repeat and bench
    kernel.Launch(grid, block, 0, params, nil)
}

func(b *CudaGenerator) FpropSoftmax(x backends.Value, axis int) backends.Value {
    // reciprocal(sum(exp(x - max(x, axis=axis)), axis=axis)) * exp(x - max(x, axis=axis))
    return b.Reciprocal(b.Sum(b.Exp(x.Sub(b.Max(x, axis))), axis)).Mul(
        b.Exp(x.Sub(b.Max(x, axis))))
}

func(b *CudaGenerator) FpropTransform(
        nglayer backends.Layer, 
        transform backends.Transform, 
        inputs backends.Tensor, 
        outputs backends.Tensor, 
        relu bool) {
    outputs.Assign(transform.Call(inputs))
}

func(b *CudaGenerator) BpropTransform(
        nglayer backends.Layer,
        transform backends.Transform,
        outputs backends.Tensor,
        errors backends.Tensor,
        deltas backends.Tensor,
        relu bool) {
    deltas.Assign(transform.Bprop(outputs).Mul(errors))
}

func(b *CudaGenerator) FpropSkipNode(x backends.Tensor, y backends.Tensor, beta float64) {
    // y[] = y * beta + x
    y.Assign(y.Mul(b.Float(beta)).Add(x))

}

func(b *CudaGenerator) BpropSkipNode(
        errors backends.Tensor, deltas backends.Tensor, alpha float64, beta float64) {
    // deltas[] = deltas * beta + alpha * errors
    deltas.Assign(deltas.Mul(b.Float(beta)).Add(b.Float(alpha).Mul(errors)))
}

// Generator methods

func(b *CudaGenerator) BuildProlog() {
    b.EnterProlog()

    b.WriteLine("")
    b.WriteLine("#include \"%shost.h\"", b.filePrefix)
    b.WriteLine("#include \"%skernels.h\"", b.filePrefix)
    b.WriteLine("")
    b.WriteLine("using namespace arhat;")
    b.WriteLine("using namespace arhat::cuda;")
    b.WriteLine("")

    if len(b.hostNamespace) != 0 {
        b.WriteLine("namespace %s {", b.hostNamespace)
        b.WriteLine("")
    }

    numBuffers := len(b.buffers)
    if numBuffers != 0 {
        b.WriteLine("void *buffers[%d];", numBuffers)
        b.WriteLine("")
    }

    b.DeclareGlobalObjects()

    b.WriteLine("void Prolog() {")
    b.Indent(1)

    b.WriteLine("RngInit();")
    b.WriteLine("CublasInit();")
    b.WriteLine("")

    rngSeed := b.RngSeed()
    if rngSeed != base.IntNone {
        b.WriteLine("RngSetSeed(%d);", rngSeed)
    }

    for _, buf := range b.buffers {
        base.Assert(buf.offset == 0)
        b.WriteLine("buffers[%d] = CudaMemAlloc(%d);", buf.index, buf.nbytes)
    }

    b.InitializeGlobalObjects()

    b.Indent(-1)
    b.WriteLine("}")
    b.WriteLine("")

    b.ExitProlog()
}

func(b *CudaGenerator) GetData(
        dest string, start string, stop string, x backends.Tensor) string {
    buf := x.(*acc.AccTensor).AccData().(*CudaDeviceAllocation)
    itemSize := x.Dtype().ItemSize()
    return fmt.Sprintf("CudaGetData(%s, %s, %s, %s, %d)", dest, start, stop, buf, itemSize)
}

func(b *CudaGenerator) GetMetricSum(x backends.Tensor, start string, stop string) string {
    buf := x.(*acc.AccTensor).AccData().(*CudaDeviceAllocation)
    bufRef := formatBufferRef(buf, true)
    return fmt.Sprintf("CudaGetFloatSum((float *)%s, %s, %s)", bufRef, start, stop)
}

func(b *CudaGenerator) OutputCode(outDir string) error {
    if b.buildMainCpp {
        if err := b.outputMainCpp(outDir); err != nil {
            return err
        }
    }
    if err := b.outputHostCu(outDir); err != nil {
        return err
    }
    if err := b.outputKernelsCu(outDir); err != nil {
        return err
    }
    if err := b.outputHostH(outDir); err != nil {
        return err
    }
    if err := b.outputKernelsH(outDir); err != nil {
        return err
    }
    return nil
}

var mainCppCode = `
#include <cstdio>
#include "%shost.h"

int main() {
    %sMain();
    return 0;
}

`

func(b *CudaGenerator) outputMainCpp(outDir string) error {
    fn := b.filePrefix + "main.cpp"
    fp, err := os.Create(path.Join(outDir, fn))
    if err != nil {
        return err
    } 
    ns := ""
    if len(b.hostNamespace) != 0 {
        ns = b.hostNamespace + "::"
    }
    fmt.Fprintf(fp, mainCppCode, b.filePrefix, ns)
    fp.Close()
    return nil
}

func(b *CudaGenerator) outputHostCu(outDir string) error {
    fn := b.filePrefix + "host.cu"
    fp, err := os.Create(path.Join(outDir, fn))
    if err != nil {
        return err
    } 
    code := strings.Join(b.HostChunks(), "")
    fmt.Fprintf(fp, "%s", code)
    if len(b.hostNamespace) != 0 {
        fmt.Fprintf(fp, "} // %s\n", b.hostNamespace)
        fmt.Fprintf(fp, "\n")
    }
    fp.Close()
    return nil
}

func(b *CudaGenerator) outputKernelsCu(outDir string) error {
    chunks := b.DeviceChunks()
    w := 0
    for n := len(chunks); n > 0; n /= 10 {
        w++
    }
    if w == 0 {
        return nil
    }
    sfxFmt := fmt.Sprintf("%%0%dd", w)

    for index, chunk := range chunks {
        sfx := fmt.Sprintf(sfxFmt, index+1)
        fn := fmt.Sprintf("%skernel%s.cu", b.filePrefix, sfx)
        fp, err := os.Create(path.Join(outDir, fn))
            if err != nil {
            return err
        } 
        fmt.Fprintf(fp, "\n")
        fmt.Fprintf(fp, "#include \"%skernels.h\"\n", b.filePrefix)
        fmt.Fprintf(fp, "\n")
        fmt.Fprintf(fp, "%s", chunk)
        fmt.Fprintf(fp, "\n")
        fp.Close()
    }
    return nil
}

var hostHCode = `
#pragma once

#include <cstdio>

#include "runtime/arhat.h"
#include "runtime_cuda/arhat.h"

void Main();

`
// end hostHCode

var hostHCodeNs = `
#pragma once

#include <cstdio>

#include "runtime/arhat.h"
#include "runtime_cuda/arhat.h"

namespace %s {

void Main();

} // %s

`
// end hostHCodeNs

func(b *CudaGenerator) outputHostH(outDir string) error {
    fn := b.filePrefix + "host.h"
    fp, err := os.Create(path.Join(outDir, fn))
    if err != nil {
        return err
    } 
    if len(b.hostNamespace) != 0 {
        fmt.Fprintf(fp, hostHCodeNs, b.hostNamespace, b.hostNamespace)
    } else {
        fmt.Fprintf(fp, hostHCode)
    }
    fp.Close()
    return nil
}

var kernelsHCompat =
`#if defined(CUDART_VERSION) && CUDART_VERSION < 9000

__device__ inline unsigned __ballot_sync(unsigned mask, int predicate) {
    return __ballot(predicate);
}

template<typename T>
__device__ T __shfl_sync(unsigned mask, T var, int srcLane, int width=warpSize) {
    return __shfl(var, srcLane, width);
}

template<typename T>
__device__ T __shfl_xor_sync(unsigned mask, T var, int laneMask, int width=warpSize) {
    return __shfl_xor(var, laneMask, width);
}

#endif
`

func(b *CudaGenerator) outputKernelsH(outDir string) error {
    fn := b.filePrefix + "kernels.h"
    fp, err := os.Create(path.Join(outDir, fn))
    if err != nil {
        return err
    } 
    fmt.Fprintf(fp, "\n")
    fmt.Fprintf(fp, "#pragma once\n")
    fmt.Fprintf(fp, "\n")
    fmt.Fprintf(fp, "%s", kernelsHCompat)
    fmt.Fprintf(fp, "\n")
    for _, chunk := range b.DeviceChunks() {
        pos := strings.Index(chunk, "__global__")
        base.Assert(pos >= 0)
        start := pos
        pos = strings.Index(chunk[pos:], ")")
        base.Assert(pos >= 0)
        sig := chunk[start:start+pos+1]
        fmt.Fprintf(fp, "%s;\n", sig)
        fmt.Fprintf(fp, "\n")
    }
    fp.Close()
    return nil
}

func(b *CudaGenerator) FormatBufferRef(tensor backends.Tensor, paren bool) string {
    accTensor := tensor.(*acc.AccTensor)
    buf := accTensor.AccData()
    cudaBuf := buf.(*CudaDeviceAllocation)
    return formatBufferRef(cudaBuf, paren)
}

// AccGenerator methods

func(b *CudaGenerator) Assign(out *acc.AccTensor, value backends.Value) backends.Value {
    optree := backends.BuildOpTreeNode(backends.Assign, out, value)

    // get post order stack
    stack := optree.Traverse(nil)

    // bypass stage creation
    if acc.IsSimpleStack(stack) {
        return b.CompoundKernel(stack)
    }

    // create stages and evaluate
    stacks := b.SplitToStacks(optree)

    for _, stack = range stacks {
        compoundDot := false
        if len(stack) == 5 {
            if n, ok := stack[3].(*backends.OpTreeNode); ok && n.Op() == backends.Dot {
                compoundDot = true
            }
        }
        if compoundDot {
            // evaluate the simple dot
            b.CompoundDot(
                stack[1].(backends.Tensor), 
                stack[2].(backends.Tensor), 
                stack[0].(backends.Tensor),
                1.0,   // alpha
                0.0,   // beta
                false, // relu
                nil)   // bsum
        } else {
            b.CompoundKernel(stack)
        }       
    }

    // TODO(orig): to be removed, used in partial
    return stacks[len(stacks)-1][0]
}

func(b *CudaGenerator) MemAlloc(nbytes int) acc.DeviceAllocation {
    index := b.nextBufferIndex
    b.nextBufferIndex++
    buf := NewCudaDeviceAllocation(index, nbytes, 0)
    b.buffers = append(b.buffers, buf)
    return buf
}

func(b *CudaGenerator) MemsetD8Async(dest acc.DeviceAllocation, data uint8, count int) {
    buf := formatBufferRef(dest.(*CudaDeviceAllocation), false)
    b.WriteLine("CudaMemsetD8Async(%s, %du, %d);", buf, data, count)
}

func(b *CudaGenerator) MemsetD16Async(dest acc.DeviceAllocation, data uint16, count int) {
    buf := formatBufferRef(dest.(*CudaDeviceAllocation), false)
    b.WriteLine("CudaMemsetD16Async(%s, %du, %d);", buf, data, count)
}

func(b *CudaGenerator) MemsetD32Async(dest acc.DeviceAllocation, data uint32, count int) {
    buf := formatBufferRef(dest.(*CudaDeviceAllocation), false)
    b.WriteLine("CudaMemsetD32Async(%s, %du, %d);", buf, data, count)
}

func(b *CudaGenerator) MemcpyDtodAsync(
        dest acc.DeviceAllocation, src acc.DeviceAllocation, size int) {
    destBuf := formatBufferRef(dest.(*CudaDeviceAllocation), false)
    srcBuf := formatBufferRef(src.(*CudaDeviceAllocation), false)
    b.WriteLine("CudaMemcpyDtodAsync(%s, %s, %d);", destBuf, srcBuf, size)
}

func(b *CudaGenerator) GetInt(src acc.DeviceAllocation, size int) string {
    ref := formatBufferRef(src.(*CudaDeviceAllocation), false)
    return fmt.Sprintf("CudaGetInt(%s, %d)", ref, size)
}

func(b *CudaGenerator) GetFloat(src acc.DeviceAllocation, size int) string {
    ref := formatBufferRef(src.(*CudaDeviceAllocation), false)
    return fmt.Sprintf("CudaGetFloat(%s, %d)", ref, size)
}

// generic compound kernel support

func(b *CudaGenerator) CompoundKernel(args []backends.Value) backends.Value {
    randState := b.GetRandStateDev()
    computeCapability := b.computeCapability

    typeArgs, kernelArgs, out, threads, blocks, reduction := 
        acc.CompileCompoundKernel(randState, args)

    key := 
        fmt.Sprintf("compound_%s_%d_%d", 
            acc.HashTypeArgs(typeArgs), computeCapability[0], computeCapability[1])

    kernel := b.LookupKernel(key)
    if kernel == nil {
        code, name := BuildCompoundKernel(typeArgs, computeCapability)
        kernel = NewKernel(b, name, code)
        b.RegisterKernel(key, kernel)
    }

    id, ok := b.kernelIdMap[kernel]
    if !ok {
        id = b.makeKernelId()
        b.kernelIdMap[kernel] = id
        code := strings.Replace(kernel.code, kernel.name, id, 1)
        code = beautifyKernel(code, true)
        // device code for kernel
        b.EnterKernel()
        b.WriteChunk(code)
        b.ExitKernel()
    }

    kernelSig := makeKernelSig(kernel.code, kernel.name)
    argList := makeKernelArgs(kernelSig, kernelArgs)

    shared := 0
    if reduction && threads > 32 {
        shared = threads * 4
    }

    // host code for kernel launch
    // SKIPPED: benchmarking
    var launchConf string
    if shared != 0 {
        launchConf = fmt.Sprintf("%d, %d, %d", blocks, threads, shared)
    } else {
        launchConf = fmt.Sprintf("%d, %d", blocks, threads)
    }
    if len(argList) <= 3 {
        b.WriteLine("%s<<<%s>>>(%s);", id, launchConf, strings.Join(argList, ", "))
    } else {
        b.WriteLine("%s<<<%s>>>(", id, launchConf)
        b.Indent(1)
        n := len(argList)
        for i := 0; i < n; i++ {
            s := argList[i]
            if i != n - 1 {
                s += ","
            } else {
                s += ");"
            }
            b.WriteLine("%s", s)
        }
        b.Indent(-1)
    }

    return out
}

// local helper methods

func(b *CudaGenerator) writeLongStmt(s string, args ...interface{}) {
    stmt := fmt.Sprintf(s, args...)
    lines := strings.Split(stmt, "#")
    b.WriteLine("%s", lines[0])
    b.Indent(1)
    for _, s := range lines[1:] {
        b.WriteLine("%s", s)
    }
    b.Indent(-1)
}

func(b *CudaGenerator) makeKernelId() string {
    index := b.nextKernelIndex
    b.nextKernelIndex++
    return fmt.Sprintf("%sKernel_%d", b.kernelPrefix, index)
}

// local helper functions

func makeKernelSig(code string, name string) []string {
    pos := strings.Index(code, name)
    base.Assert(pos >= 0)
    start := pos + len(name)
    pos = strings.Index(code[start:], "(")
    base.Assert(pos >= 0)
    start += pos + 1
    pos = strings.Index(code[start:], ")")
    base.Assert(pos >= 0)
    sig := strings.Split(code[start:start+pos], ",")
    // strip argument names
    mustSkip := func(c byte) bool {
        return ((c >= 'A' && c <= 'Z') ||
            (c >= 'a' && c <= 'z') ||
            (c >= '0' && c <= '9') || c == '_')
    }
    n := len(sig)
    for i := 0; i < n; i++ {
        s := strings.Trim(sig[i], " \n")
        k := len(s) - 1
        for k >= 0 {
            if !mustSkip(s[k]) {
                break
            }
            k--
        }
        base.Assert(k >= 0)
        sig[i] = strings.TrimRight(s[:k+1], " \n")
    }
    return sig
}

func makeKernelArgs(kernelSig []string, kernelArgs []acc.KernelArgument) []string {
    n := len(kernelArgs)
    base.Assert(len(kernelSig) == n)
    result := make([]string, n)
    for i := 0; i < n; i++ {
        switch v := kernelArgs[i].(type) {
        case bool:
            if v {
                result[i] = "true"
            } else {
                result[i] = "false"
            }
        case int:
            result[i] = fmt.Sprintf("%d", v)
        case float64:
            result[i] = formatFloat32(v)
        case string:
            // symbol
            result[i] = v
        case nil:
            // support for nil tensors/buffers
            result[i] = "NULL"
        case *CudaDeviceAllocation:
            result[i] = formatBufferArg(kernelSig[i], v)
        case *acc.AccTensor:
            result[i] = formatBufferArg(kernelSig[i], v.AccData().(*CudaDeviceAllocation))
        default:
            base.ValueError("Argument %d: invalid type: %t", i, kernelArgs[i])
        }
    }
    return result
}

func formatBufferArg(sig string, buf *CudaDeviceAllocation) string {
    if strings.HasPrefix(sig, "const") {
        sig = strings.TrimLeft(sig[5:], " \n")
    }
    if strings.HasSuffix(sig, "__restrict__") {
        sig = strings.TrimRight(sig[:len(sig)-12], " \n")
    }
    return fmt.Sprintf("(%s)%s", sig, formatBufferRef(buf, true))
}

func formatFloat32(x float64) string {
    return generators.FormatFloat32(x)
}

func formatBufferRef(buf *CudaDeviceAllocation, paren bool) string {
    if buf.offset != 0 {
        if (paren) {
            return fmt.Sprintf("((uint8_t *)buffers[%d]+%d)", buf.index, buf.offset)
        } else {
            return fmt.Sprintf("(uint8_t *)buffers[%d]+%d", buf.index, buf.offset)
        }
    } else {
        return fmt.Sprintf("buffers[%d]", buf.index)
    }
}

//
//    Minimum beautify procedure for legacy kernels
//

func beautifyKernel(code string, autoIndent bool) string {
    var result []string
    emptyLine := false
    indent := 0
    lines := strings.Split(code, "\n")
    for _, line := range lines {
        content := strings.Trim(line, " \t")
        if len(content) == 0 {
            if !emptyLine {
                result = append(result, "")
                emptyLine = true
            }
        } else {
            if autoIndent {
                lb := strings.Contains(line, "{")
                rb := strings.Contains(line, "}")
                if rb && !lb {
                    indent--
                }
                if indent != 0 {
                    // retain layout at top level
                    line = strings.Repeat(" ", 4*indent) + content
                }
                if lb && !rb {
                    indent++
                }
            }
            result = append(result, line)
            emptyLine = false
        }
    }
    if !emptyLine {
        // code doesn't end with '\n'
        result = append(result, "")
    }
    result = append(result, "")
    return strings.Join(result, "\n")
}

