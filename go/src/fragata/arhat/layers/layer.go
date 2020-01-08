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

package layers

import (
    "fmt"
    "fragata/arhat/backends"
    "fragata/arhat/base"
    "fragata/arhat/initializers"
    "fragata/arhat/transforms"
    "strings"
)

//
//    local functions
//

// Helper function to interpret the tensor layout of preceding layer to handle
// non-recurrent, recurrent, and local layers.
func interpretInShape(xshape []int) []int {
    switch len(xshape) {
    case 1:
        return []int{xshape[0], 1}
    case 2:
        return xshape
    default:
        return []int{base.IntsProd(xshape), 1}
    }
}

func assertSingle(x []backends.Tensor) {
    base.Assert(len(x) <= 1) // nil is allowed
}

//
//    States
//

// wrapped in structure so that optimizer can update states remotely via Param

type States struct {
    states []backends.Tensor
}

func NewStates() *States {
    return new(States)
}

func(s *States) Get() []backends.Tensor {
    return s.states
}

func(s *States) Set(states []backends.Tensor) {
    s.states = states
}

//
//    Param
//

type Param struct {
    w backends.Tensor
    dw backends.Tensor
    states *States
}

func(p *Param) W() backends.Tensor {
    return p.w
}

func(p *Param) Dw() backends.Tensor {
    return p.dw
}

func(p *Param) States() []backends.Tensor {
    return p.states.Get()
}

func(p *Param) SetStates(states []backends.Tensor) {
    p.states.Set(states)
}

//
//    accParam
//

type accParam struct {
    accP backends.Tensor
    p backends.Tensor
}

//
//    ParamReader
//

type ParamReader interface {
    Read(x backends.Tensor)
}

//
//    ParamWriter
//

type ParamWriter interface {
    Write(x backends.Tensor)
}

//
//    LayerItem
//

type LayerItem interface {
    Atom() Layer
    List() []Layer
}

//
//    InputObject
//

//
//    This structure encapsulates all variants of valid information about
//    layer input streams.

//    These include:
//        [1] for single input stream: 
//            - shape specification (int, []int)
//            - tensor (Tensor)
//            - layer (Layer) with single stream output and
//        [2] for multiple input streams:
//            - array of single input streams ([]InputObject)
//            - layer (Layer) with multiple stream output
//
//    Shape specifications represent tensors produced by data iterators
//    Tensors represent tensor batches
//    Layers represent tensors produced as layer output
//
//    This design is quite ugly but at least it provides simple and straightforward
//    way to keep all valid variant specifications in one place. It can be improved
//    in the future by designing better abstractions and reducing number of variants
//    based on more thorough study of all use scenarios.
//

type InputObject struct {
    obj interface{}
}

func MakeInputObject(obj interface{}) InputObject {
    switch obj.(type) {
    case nil, int, []int, backends.Tensor, Layer:
        // ok
    case []InputObject:
        v := obj.([]InputObject)
        for _, o := range v {
            o.AssertSingle()
        }
    default:
        base.AssertionError("Invalid input object type")
    }
    return InputObject{obj}
}

func(o InputObject) IsNil() bool {
    return (o.obj == nil)
}

func(o InputObject) Shape() []int {
    var shape []int
    switch v := o.obj.(type) {
    case int:
        shape = []int{v}
    case []int:
        shape = v
    case backends.Tensor:
        shape = v.Shape()
        shape = []int{shape[0], shape[1]/backends.Be().Bsz()}
    default:
        base.AssertionError("Invalid input object type")
    }
    return shape
}

func(o InputObject) MultiShape() [][]int {
    var result [][]int
    switch v := o.obj.(type) {
    case Layer:
        result = v.MultiOutShape()
    case []InputObject:
        result := make([][]int, len(v))
        for i, obj := range v {
            result[i] = obj.Shape()
        }
    }
    return result
}

func(o InputObject) Layer() Layer {
    if v, ok := o.obj.(Layer); ok {
        return v
    }
    return nil
}

func(o InputObject) AssertSingle() {
    multi := false
    switch v := o.obj.(type) {
    case Layer:
        multi = (v.MultiOutShape() != nil)
    case []InputObject:
        multi = true
    }
    if multi {
        base.AssertionError("Multistream input is not allowed")
    }
}

//
//    Layer
//

type Layer interface {
    base.Object
    Outputs() []backends.Tensor
    HasParams() bool
    OwnsOutput() bool
    Deltas() []backends.Tensor
    PrevLayer() Layer
    InShape() []int
    OutShape() []int
    MultiOutShape() [][]int
    String() string
    NestedStr(level int) string
    Configure(inObj InputObject)
    Allocate(sharedOutputs backends.Tensor, accumulateUpdates bool)
    AllocateDeltas(globalDeltas *DeltasTree)
    SetDeltas(deltaBuffers *DeltasTree)
    SetNext(layer Layer)
    GetParams() []Param
    Fprop(inputs []backends.Tensor, inference bool, beta float64) []backends.Tensor
    Bprop(errors []backends.Tensor, alpha float64, beta float64) []backends.Tensor
    GetTerminal() []Layer
    NestDeltas() bool
    BatchSum() backends.Tensor
    ReadParams(r ParamReader)
    WriteParams(w ParamWriter)
}

//
//    LayerBase
//

type LayerBase struct {
    base.ObjectBase
    outputs []backends.Tensor
    hasParams bool
    inputs []backends.Tensor
    ownsOutput bool
    ownsDelta bool
    deltas []backends.Tensor
    nextLayer Layer
    actualBsz int
    actualSeqLen int
    accOn bool
    prevLayer Layer
    inShape []int
    outShape []int
}

var layerBaseInitArgMap = base.ArgMap{
    "name": base.NewAnyArgOpt(""), // passthru
}

func(n *LayerBase) Init(self base.Object, args base.Args) {
    args = layerBaseInitArgMap.Expand(args)
    n.ObjectBase.Init(self, args.Filter([]string{"name"}))
    n.outputs = nil
    n.hasParams = false
    n.inputs = nil
    n.ownsOutput = true
    n.ownsDelta = false
    n.deltas = nil
    n.nextLayer = nil
    n.actualBsz = base.IntNone
    n.actualSeqLen = base.IntNone
    n.accOn = false
}

func(n *LayerBase) Outputs() []backends.Tensor {
    return n.outputs
}

func(n *LayerBase) HasParams() bool {
    return n.hasParams
}

func(n *LayerBase) OwnsOutput() bool {
    return n.ownsOutput
}

func(n *LayerBase) Deltas() []backends.Tensor {
    return n.deltas
}

func(n *LayerBase) PrevLayer() Layer {
    return n.prevLayer
}

func(n *LayerBase) InShape() []int {
    return n.inShape
}

func(n *LayerBase) OutShape() []int {
    return n.outShape
}

func(n *LayerBase) MultiOutShape() [][]int {
    // used for Tree and similar layers with multiple output streams
    return nil
}

func(n *LayerBase) String() string {
    return fmt.Sprintf("%s %s", n.Self().ShortClassName(), n.Name())
}

func(n *LayerBase) NestedStr(level int) string {
    return strings.Repeat("  ", level) + n.ToLayer().String()
}

func(n *LayerBase) Configure(inObj InputObject) {
    // layers with multiple output streams must overload this method
    inObj.AssertSingle()
    if v := inObj.Layer(); v != nil {
        n.prevLayer = v
        n.inShape = v.OutShape()
        return
    }
    n.prevLayer = nil
    n.inShape = inObj.Shape()
}

func(n *LayerBase) Allocate(sharedOutputs backends.Tensor, accumulateUpdates bool) {
    if n.outputs != nil {
        return
    }
    if n.ownsOutput {
        output := backends.Be().Iobuf(n.outShape, nil, base.DtypeNone, "", true, sharedOutputs)
        n.outputs = []backends.Tensor{output}
    }
}

func(n *LayerBase) AllocateDeltas(globalDeltas *DeltasTree) {
    globalDeltas.ProcLayer(n.ToLayer())
}

func(n *LayerBase) SetDeltas(deltaBuffers *DeltasTree) {
    if n.ownsDelta && n.prevLayer != nil {
        if toBranchNode(n.prevLayer) != nil {
            n.deltas = n.prevLayer.Deltas()
        } else {
            delta := 
                backends.Be().Iobuf(
                    n.inShape, nil, base.DtypeNone, "", true, deltaBuffers.buffers[0])
            n.deltas = []backends.Tensor{delta}
            deltaBuffers.ReverseBuffers()
        }
    } else {
        n.deltas = nil
    }
}

func(n *LayerBase) SetNext(layer Layer) {
    n.nextLayer = layer
}

func(n *LayerBase) GetParams() []Param {
    // defined only to satisfy Layer interface
    // overloaded for parameterized layers
    //     and must be called only for such layers
    base.AssertionError("GetParams can be called for parameter layers only")
    return nil
}

func(n *LayerBase) GetTerminal() []Layer {
    return []Layer{n.ToLayer()}
}

func(n *LayerBase) ToLayer() Layer {
    return n.Self().(Layer)
}

// LayerItem methods

func(n *LayerBase) Atom() Layer { 
    return n.ToLayer()
}

func(n *LayerBase) List() []Layer {
    return nil
}

func(n *LayerBase) NestDeltas() bool {
    return false
}

func(n *LayerBase) BatchSum() backends.Tensor {
    return nil
}

//
//    SKIPPED: SetAccOn method and Accumulates (AccumPre, AccumPost) wrapper
//        Apparently they are used only for GAN models which, for the time being,
//        are out of scope
//

func(n *LayerBase) ReadParams(r ParamReader) {
    // nothing to do by default 
    // to be overloaded for layers with parameters
}

func(n *LayerBase) WriteParams(w ParamWriter) {
    // nothing to do by default 
    // to be overloaded for layers with parameters
}

//
//    BranchNode
//

type BranchNode struct {
    LayerBase
}

var branchNodeInstances = make(map[string]*BranchNode)

var branchNodeInitArgMap = base.ArgMap{
    "name": base.NewStringArgOpt(""),
}

func NewBranchNode(args ...interface{}) *BranchNode {
    a := base.MakeArgs(args)
    if name, ok := a["name"].(string); ok {
        if n, ok := branchNodeInstances[name]; ok {
            return n
        }
    }
    n := new(BranchNode)
    n.Init(n, a)
    return n
}

func(n *BranchNode) Init(self base.Object, args base.Args) {
    args = branchNodeInitArgMap.Expand(args)
    name := args["name"].(string)
    if _, ok := branchNodeInstances[name]; ok {
        return
    }
    branchNodeInstances[name] = n
    n.LayerBase.Init(self, args.Filter([]string{"name"}))
    n.ownsOutput = false
}

func(n *BranchNode) ClassName() string {
    return "arhat.layers.BranchNode"
}

func(n *BranchNode) Configure(inObj InputObject) {
    inObj.AssertSingle()
    if n.inShape != nil && inObj.IsNil() {
        return // previously configured, so just return
    }
    n.LayerBase.Configure(inObj)
    n.outShape = n.inShape
}

func(n *BranchNode) SetDeltas(deltaBuffers *DeltasTree) {
    if n.deltas == nil {
        delta := 
            backends.Be().Iobuf(
                n.inShape, nil, base.DtypeNone, "", true, deltaBuffers.buffers[0])
        n.deltas = []backends.Tensor{delta}
        deltaBuffers.ReverseBuffers()
    }
}

func(n *BranchNode) Fprop(
        inputs []backends.Tensor, inference bool, beta float64) []backends.Tensor {
    assertSingle(inputs)
    if n.outputs == nil && inputs != nil {
        n.outputs = inputs
    }
    return n.outputs
}

func(n *BranchNode) Bprop(
        errors []backends.Tensor, alpha float64, beta float64) []backends.Tensor {
    // nothing to do
    return nil
}

//
//    SkipNode
//

type SkipNode struct {
    LayerBase
}

var skipNodeInitArgMap = base.ArgMap{
    "name": base.NewAnyArgOpt(""), // passthru
}

func NewSkipNode(args ...interface{}) *SkipNode {
    n := new(SkipNode)
    n.Init(n, base.MakeArgs(args))
    return n
}

func(n *SkipNode) Init(self base.Object, args base.Args) {
    args = skipNodeInitArgMap.Expand(args)
    n.LayerBase.Init(self, args.Filter([]string{"name"}))
    n.ownsDelta = true
}

func(n *SkipNode) ClassName() string {
    return "arhat.layers.SkipNode"
}

func(n *SkipNode) Configure(inObj InputObject) {
    inObj.AssertSingle()
    n.LayerBase.Configure(inObj)
    n.outShape = n.inShape
}

func(n *SkipNode) Fprop(
        inputs []backends.Tensor, inference bool, beta float64) []backends.Tensor {
    assertSingle(inputs)
    be := backends.Be()
    be.FpropSkipNode(inputs[0], n.outputs[0], beta)
    return n.outputs
}

func(n *SkipNode) Bprop(
        errors []backends.Tensor, alpha float64, beta float64) []backends.Tensor {
    assertSingle(errors)
    be := backends.Be()
    be.BpropSkipNode(errors[0], n.deltas[0], alpha, beta)
    return n.deltas 
}

//
//    Pooling
//

var poolingOpEnum = base.EnumDef{
    "max": int(backends.PoolOpMax),
    "avg": int(backends.PoolOpAvg),
    // TODO: Add "l2" => PoolOpL2 ?
}

type Pooling struct {
    LayerBase
    poolParams backends.PoolParams
    op backends.PoolOp
    // fshape
    t int
    r int 
    s int
    // strides
    strH int
    strW int
    // padding
    padH int
    padW int
    nglayer backends.PoolLayer
    argmax backends.Tensor
}

var poolingInitArgMap = base.ArgMap{
    "fshape": base.NewAnyArg(),
    "op": base.NewEnumArgOpt(poolingOpEnum, "max"),
    "strides": base.NewAnyArgOpt(nil),
    "padding": base.NewAnyArgOpt(nil),
    "name": base.NewAnyArgOpt(""), // passthru
}

var poolingInitStridesMap = base.ArgMap{
    "str_h": base.NewIntArgOpt(base.IntNone),
    "str_w": base.NewIntArgOpt(base.IntNone),
}

var poolingInitPaddingMap = base.ArgMap{
    "pad_h": base.NewIntArgOpt(base.IntNone),
    "pad_w": base.NewIntArgOpt(base.IntNone),
}

func NewPooling(args ...interface{}) *Pooling {
    n := new(Pooling)
    n.Init(n, base.MakeArgs(args))
    return n
}

func(n *Pooling) Init(self base.Object, args base.Args) {
    args = poolingInitArgMap.Expand(args)
    n.LayerBase.Init(self, args.Filter([]string{"name"}))
    op := backends.PoolOp(args["op"].(int))
    n.poolParams.Init(op)
    n.op = op
    n.t = base.IntNone
    n.r = base.IntNone
    n.s = base.IntNone
    switch v := args["fshape"].(type) {
    case int:
        n.r = v
        n.s = v
    case []int:
        switch len(v) {
        case 2:
            n.r = v[0]
            n.s = v[1]
        case 3:
            n.t = v[0] 
            n.r = v[1]
            n.s = v[2]
        default:
            base.InvalidArgument("fshape")
        }
    case string:
        if v != "all" {
            base.InvalidArgument("fshape")
        }
    }
    n.strH = base.IntNone
    n.strW = base.IntNone
    switch v := args["strides"].(type) {
    case nil:
        // ok
    case int:
        n.strH = v
        n.strW = v
    case base.Args:
        w := poolingInitStridesMap.Expand(v)
        n.strH = w["str_h"].(int)
        n.strW = w["str_w"].(int)
    default:
        base.InvalidArgument("strides")
    }
    n.padH = base.IntNone
    n.padW = base.IntNone
    switch v := args["padding"].(type) {
    case nil:
        // ok
    case int:
        n.padH = v
        n.padW = v
    case base.Args:
        w := poolingInitPaddingMap.Expand(v)
        n.padH = w["pad_h"].(int)
        n.padW = w["pad_w"].(int)
    default:
        base.InvalidArgument("padding")
    }
    n.ownsDelta = true
    n.poolParams.T = n.t
    n.poolParams.R = n.r
    n.poolParams.S = n.s
    n.poolParams.StrH = n.strH
    n.poolParams.StrW = n.strW
    n.poolParams.PadH = n.padH
    n.poolParams.PadW = n.padW
    n.nglayer = nil
}

func(n *Pooling) ClassName() string {
    return "arhat.layers.Pooling"
}

func(n *Pooling) Configure(inObj InputObject) {
    inObj.AssertSingle()
    n.LayerBase.Configure(inObj)
    if n.nglayer == nil {
        be := backends.Be()
        switch len(n.inShape) {
        case 3:
            n.poolParams.C = n.inShape[0]
            n.poolParams.H = n.inShape[1]
            n.poolParams.W = n.inShape[2]
        case 4:
            n.poolParams.C = n.inShape[0]
            n.poolParams.D = n.inShape[1]
            n.poolParams.H = n.inShape[2]
            n.poolParams.W = n.inShape[3]
        default:
            base.Assert(false)
        }
        n.poolParams.N = be.Bsz()
        if n.poolParams.R == base.IntNone {
            n.poolParams.R = n.poolParams.H
            n.poolParams.S = n.poolParams.W
        }
        n.nglayer = be.NewPoolLayer(be.DefaultDtype(), &n.poolParams)
        dimO := n.nglayer.DimO()
        k := dimO[0]
        m := dimO[1]
        p := dimO[2]
        q := dimO[3]
        if len(n.inShape) == 3 {
            n.outShape = []int{k, p, q}
        } else { // 4
            n.outShape = []int{k, m, p, q}
        }
    }
}

func(n *Pooling) SetDeltas(deltaBuffers *DeltasTree) {
    n.LayerBase.SetDeltas(deltaBuffers)
    if n.op == backends.PoolOpMax {
        n.argmax = backends.Be().NewTensor(n.outputs[0].Shape(), base.Uint8)
    } else {
        n.argmax = nil
    }
}

func(n *Pooling) Fprop(
        inputs []backends.Tensor, inference bool, beta float64) []backends.Tensor {
    assertSingle(inputs)
    be := backends.Be()
    n.inputs = inputs
    be.FpropPool(
        n.nglayer,    // layer
        inputs[0],    // i
        n.outputs[0], // o
        n.argmax,     // argmax
        1.0,          // alpha
        beta)         // beta
    return n.outputs
}

func(n *Pooling) Bprop(
        errors []backends.Tensor, alpha float64, beta float64) []backends.Tensor {
    assertSingle(errors)
    be := backends.Be()
    be.BpropPool(
        n.nglayer,   // layer
        errors[0],   // i
        n.deltas[0], // o
        n.argmax,    // argmax
        alpha,       // alpha
        beta)        // beta
    return n.deltas
}

//
//    ParameterLayer
//

//
//    ACHTUNG: Fields accumulateUpdates, accDw, and accParams are apparently used
//        for GAN networks only which, for the time being, are out of scope.
//        Support for these fields at present phase can be removed.
//

type ParameterLayer struct {
    LayerBase
    init initializers.Initializer
    w backends.Tensor
    dw backends.Tensor
    weightShape []int
    batchSum backends.Tensor
    batchSumShape []int
    states *States
    accumulateUpdates bool
    accDw backends.Tensor
    accParams []accParam
    initParams func(shape []int)
}

var parameterLayerInitArgMap = base.ArgMap{
    "init": initializers.NewInitializerArgOpt(nil),
    "name": base.NewAnyArgOpt(""), // passthru
}

func(n *ParameterLayer) Init(self base.Object, args base.Args) {
    args = parameterLayerInitArgMap.Expand(args)
    n.LayerBase.Init(self, args.Filter([]string{"name"}))
    n.hasParams = true
    n.init = initializers.ToInitializer(args["init"])
    n.w = nil
    n.dw = nil
    n.weightShape = nil
    n.batchSum = nil
    n.batchSumShape = nil
    n.states = NewStates()
    n.ownsDelta = true
    n.initParams = n.InitParams
}

func(n *ParameterLayer) Allocate(sharedOutputs backends.Tensor, accumulateUpdates bool) {
    n.LayerBase.Allocate(sharedOutputs, false)
    n.accumulateUpdates = accumulateUpdates
    if n.w == nil {
        // InitParams may be overloaded by some layers
        n.initParams(n.weightShape)
    }
    if n.batchSumShape != nil {
        n.batchSum = backends.Be().NewTensor(n.batchSumShape, base.Float32)
    }
}

func(n *ParameterLayer) InitParams(shape []int) {
    be := backends.Be()
    n.w = be.NewTensor(shape, base.DtypeNone)
    n.dw = be.NewTensor(shape, base.DtypeNone)
    n.states.Set(nil)

    // SKIPPED: Case when n.init is Tensor or Array (not allowed in this implementation)
    n.init.Fill(n.w)

    if n.accumulateUpdates {
        n.accDw = be.NewTensor(shape, base.DtypeNone)
        n.accParams = []accParam{accParam{n.accDw, n.dw}}
    }
}

func(n *ParameterLayer) GetParams() []Param {
    return []Param{Param{n.w, n.dw, n.states}}
}

func(n *ParameterLayer) BatchSum() backends.Tensor {
    return n.batchSum
}

func(n *ParameterLayer) ReadParams(r ParamReader) {
    r.Read(n.w)
}

func(n *ParameterLayer) WriteParams(w ParamWriter) {
    w.Write(n.w)
}

//
//    Convolution
//

type Convolution struct {
    ParameterLayer
    bsum bool
    convParams backends.ConvParams
    // fshape
    t int
    r int
    s int
    k int
    // strides
    strH int
    strW int
    // padding
    padH int
    padW int
    // dilation
    dilH int
    dilW int
    nglayer backends.ConvLayer
    batchSumShape []int
}

var convolutionInitArgMap = base.ArgMap{
    "fshape": base.NewAnyArg(),
    "strides": base.NewAnyArgOpt(nil),
    "padding": base.NewAnyArgOpt(nil),
    "dilation": base.NewAnyArgOpt(nil),
    "init": base.NewAnyArgOpt(nil),           // passthru
    "bsum": base.NewBoolArgOpt(false),
    "name": base.NewAnyArgOpt(""),            // passthru
}

var convolutionInitStridesMap = base.ArgMap{
    "str_h": base.NewIntArgOpt(base.IntNone),
    "str_w": base.NewIntArgOpt(base.IntNone),
}

var convolutionInitPaddingMap = base.ArgMap{
    "pad_h": base.NewIntArgOpt(base.IntNone),
    "pad_w": base.NewIntArgOpt(base.IntNone),
}

var convolutionInitDilationMap = base.ArgMap{
    "dil_h": base.NewIntArgOpt(base.IntNone),
    "dil_w": base.NewIntArgOpt(base.IntNone),
}

func NewConvolution(args ...interface{}) *Convolution {
    n := new(Convolution)
    n.Init(n, base.MakeArgs(args))
    return n
}

func(n *Convolution) Init(self base.Object, args base.Args) {
    args = convolutionInitArgMap.Expand(args)
    n.ParameterLayer.Init(self, args.Filter([]string{"init", "name"}))
    n.nglayer = nil
    n.bsum = args["bsum"].(bool)
    n.convParams.Init()
    n.t = base.IntNone
    n.r = base.IntNone
    n.s = base.IntNone
    n.k = base.IntNone
    switch v := args["fshape"].(type) {
    case nil:
        // ok
    case []int:
        switch len(v) {
        case 3:
            n.r = v[0]
            n.s = v[1]
            n.k = v[2]
        case 4:
            n.t = v[0]
            n.r = v[1]
            n.s = v[2]
            n.k = v[3]
        default:
            base.InvalidArgument("fshape")
        }
    default:
        base.InvalidArgument("fshape")
    }
    n.strH = base.IntNone
    n.strW = base.IntNone
    switch v := args["strides"].(type) {
    case nil:
        // ok
    case int:
        n.strH = v
        n.strW = v
    case base.Args:
        w := convolutionInitStridesMap.Expand(v)
        n.strH = w["str_h"].(int)
        n.strW = w["str_w"].(int)
    default:
        base.InvalidArgument("strides")
    }
    n.padH = base.IntNone
    n.padW = base.IntNone
    switch v := args["padding"].(type) {
    case nil:
        // ok
    case int:
        n.padH = v
        n.padW = v
    case base.Args:
        w := convolutionInitPaddingMap.Expand(v)
        n.padH = w["pad_h"].(int)
        n.padW = w["pad_w"].(int)
    default:
        base.InvalidArgument("padding")
    }
    n.dilH = base.IntNone
    n.dilW = base.IntNone
    switch v := args["dilation"].(type) {
    case nil:
        // ok
    case int:
        n.dilH = v
        n.dilW = v
    case base.Args:
        w := convolutionInitDilationMap.Expand(v)
        n.dilH = w["pad_h"].(int)
        n.dilW = w["pad_w"].(int)
    default:
        base.InvalidArgument("dilation")
    }
    n.convParams.T = n.t
    n.convParams.R = n.r
    n.convParams.S = n.s
    n.convParams.K = n.k
    n.convParams.StrH = n.strH
    n.convParams.StrW = n.strW
    n.convParams.PadH = n.padH
    n.convParams.PadW = n.padW
    n.convParams.DilH = n.dilH
    n.convParams.DilW = n.dilW
}

func(n *Convolution) ClassName() string {
    return "arhat.layers.Convolution"
}

func(n *Convolution) String() string {
    fmtShape := func(x []int) string {
        s := fmt.Sprintf("%d x (", x[0])
        for i, v := range x[1:] {
            if i != 0 {
                s += "x"
            }
            s += fmt.Sprintf("%d", v)
        }
        s += ")"
        return s
    }

    var fmtPad, fmtStr, fmtDil string
    a := &n.convParams
    if len(n.inShape) == 3 {
        fmtPad = fmt.Sprintf("%d,%d", a.PadH, a.PadW)
        fmtStr = fmt.Sprintf("%d,%d", a.StrH, a.StrW)
        fmtDil = fmt.Sprintf("%d,%d", a.DilH, a.DilW)
    } else {
        fmtPad = fmt.Sprintf("%d,%d,%d", a.PadD, a.PadH, a.PadW)
        fmtStr = fmt.Sprintf("%d,%d,%d", a.StrD, a.StrH, a.StrW)
        fmtDil = fmt.Sprintf("%d,%d,%d", a.DilD, a.DilH, a.DilW)
    }

    return fmt.Sprintf(
        "Convolution Layer '%s': %s inputs, %s outputs, %s padding, %s stride, %s dilation",
            n.Name(), fmtShape(n.inShape), fmtShape(n.outShape), fmtPad, fmtStr, fmtDil)
}

func(n *Convolution) Configure(inObj InputObject) {
    inObj.AssertSingle()
    n.ParameterLayer.Configure(inObj)
    if n.nglayer == nil {
        be := backends.Be()
        switch len(n.inShape) {
        case 3:
            n.convParams.C = n.inShape[0]
            n.convParams.H = n.inShape[1]
            n.convParams.W = n.inShape[2]
        case 4:
            n.convParams.C = n.inShape[0]
            n.convParams.D = n.inShape[1]
            n.convParams.H = n.inShape[2]
            n.convParams.W = n.inShape[3]
        default:
            base.Assert(false)
        }
        n.convParams.N = be.Bsz()        
        n.nglayer = be.NewConvLayer(be.DefaultDtype(), &n.convParams)
        dimO := n.nglayer.DimO()
        k := dimO[0]
        m := dimO[1]
        p := dimO[2]
        q := dimO[3]
        if m == 1 {
            n.outShape = []int{k, p, q}
        } else {
            n.outShape = []int{k, m, p, q}
        }
    }
    if n.weightShape == nil {
         n.weightShape = n.nglayer.DimF2() // (C * R * S, K)
    }
    if n.bsum {
        n.batchSumShape = []int{n.nglayer.K(), 1}
    }
}

func(n *Convolution) Fprop(
        inputs []backends.Tensor, inference bool, beta float64) []backends.Tensor {
    assertSingle(inputs)
    be := backends.Be()
    n.inputs = inputs
    be.FpropConv(
        n.nglayer,     // layer
        inputs[0],     // i
        n.w,           // f
        n.outputs[0],  // o
        nil,           // x
        nil,           // bias
        n.batchSum,    // bsum
        1.0,           // alpha
        beta,          // beta
        false,         // relu
        false,         // brelu
        0.0)           // slope
    return n.outputs
}

// SKIPPED: @Layer.accumulates decorator in original code
//     This is 'accumulates' wrapper that manages accParam objects for layers
//     that use them - required for GAN networks only, not relevant at this stage
func(n *Convolution) Bprop(
        errors []backends.Tensor, alpha float64, beta float64) []backends.Tensor {
    assertSingle(errors)
    be := backends.Be()
    if n.deltas != nil {
        be.BpropConv(
            n.nglayer,   // layer
            n.w,         // f
            errors[0],   // e
            n.deltas[0], // gradI
            nil,         // x
            nil,         // bias
            nil,         // bsum
            alpha,       // alpha
            beta,        // beta
            false,       // relu
            false,       // brelu
            0.0)         // slope
    }
    be.UpdateConv(
        n.nglayer,   // layer
        n.inputs[0], // i
        errors[0],   // e
        n.dw,        // gradF
        1.0,         // alpha
        0.0,         // beta,
        nil)         // gradBias
    return n.deltas
}

//
//    SKIPPED: ConvolutionBias
//

//
//    Deconvolution
//

type Deconvolution struct {
    ParameterLayer
    bsum bool
    deconvParams backends.DeconvParams
    // fshape
    t int
    r int
    s int
    c int
    // strides
    strH int
    strW int
    // padding
    padH int
    padW int
    // dilation
    dilH int
    dilW int
    nglayer backends.DeconvLayer
}

var deconvolutionInitArgMap = base.ArgMap{
    "fshape": base.NewAnyArg(),
    "strides": base.NewAnyArgOpt(nil),
    "padding": base.NewAnyArgOpt(nil),
    "dilation": base.NewAnyArgOpt(nil),
    "init": base.NewAnyArgOpt(nil),     // passthru
    "bsum": base.NewBoolArgOpt(false),
    "name": base.NewAnyArgOpt(""),      // passthru
}

var deconvolutionInitStridesMap = base.ArgMap{
    "str_h": base.NewIntArgOpt(base.IntNone),
    "str_w": base.NewIntArgOpt(base.IntNone),
}

var deconvolutionInitPaddingMap = base.ArgMap{
    "pad_h": base.NewIntArgOpt(base.IntNone),
    "pad_w": base.NewIntArgOpt(base.IntNone),
}

var deconvolutionInitDilationMap = base.ArgMap{
    "dil_h": base.NewIntArgOpt(base.IntNone),
    "dil_w": base.NewIntArgOpt(base.IntNone),
}

func NewDeconvolution(args ...interface{}) *Deconvolution {
    n := new(Deconvolution)
    n.Init(n, base.MakeArgs(args))
    return n
}

func(n *Deconvolution) Init(self base.Object, args base.Args) {
    args = deconvolutionInitArgMap.Expand(args)
    n.ParameterLayer.Init(self, args.Filter([]string{"init", "name"}))
    n.nglayer = nil
    n.bsum = args["bsum"].(bool)
    n.deconvParams.Init()
    n.t = base.IntNone
    n.r = base.IntNone
    n.s = base.IntNone
    n.c = base.IntNone
    switch v := args["fshape"].(type) {
    case nil:
        // ok
    case []int:
        switch len(v) {
        case 3:
            n.r = v[0]
            n.s = v[1]
            n.c = v[2]
        case 4:
            n.t = v[0]
            n.r = v[1]
            n.s = v[2]
            n.c = v[3]
        default:
            base.InvalidArgument("fshape")
        }
    default:
        base.InvalidArgument("fshape")
    }
    n.strH = base.IntNone
    n.strW = base.IntNone
    switch v := args["strides"].(type) {
    case nil:
        // ok
    case int:
        n.strH = v
        n.strW = v
    case base.Args:
        w := deconvolutionInitStridesMap.Expand(v)
        n.strH = w["str_h"].(int)
        n.strW = w["str_w"].(int)
    default:
        base.InvalidArgument("strides")
    }
    n.padH = base.IntNone
    n.padW = base.IntNone
    switch v := args["padding"].(type) {
    case nil:
        // ok
    case int:
        n.padH = v
        n.padW = v
    case base.Args:
        w := deconvolutionInitPaddingMap.Expand(v)
        n.padH = w["pad_h"].(int)
        n.padW = w["pad_w"].(int)
    default:
        base.InvalidArgument("padding")
    }
    n.dilH = base.IntNone
    n.dilW = base.IntNone
    switch v := args["dilation"].(type) {
    case nil:
        // ok
    case int:
        n.dilH = v
        n.dilW = v
    case base.Args:
        w := convolutionInitDilationMap.Expand(v)
        n.dilH = w["pad_h"].(int)
        n.dilW = w["pad_w"].(int)
    default:
        base.InvalidArgument("dilation")
    }
    n.deconvParams.T = n.t
    n.deconvParams.R = n.r
    n.deconvParams.S = n.s
    n.deconvParams.C = n.c
    n.deconvParams.StrH = n.strH
    n.deconvParams.StrW = n.strW
    n.deconvParams.PadH = n.padH
    n.deconvParams.PadW = n.padW
    n.deconvParams.DilH = n.dilH
    n.deconvParams.DilW = n.dilW
} 

func(n *Deconvolution) ClassName() string {
    return "arhat.layers.Deconvolution"
}

func(n *Deconvolution) String() string {
    fmtShape := func(x []int) string {
        s := fmt.Sprintf("%d x (", x[0])
        for i, v := range x[1:] {
            if i != 0 {
                s += "x"
            }
            s += fmt.Sprintf("%d", v)
        }
        s += ")"
        return s
    }

    var fmtPad, fmtStr, fmtDil string
    a := &n.deconvParams
    if len(n.inShape) == 3 {
        fmtPad = fmt.Sprintf("%d,%d", a.PadH, a.PadW)
        fmtStr = fmt.Sprintf("%d,%d", a.StrH, a.StrW)
        fmtDil = fmt.Sprintf("%d,%d", a.DilH, a.DilW)
    } else {
        fmtPad = fmt.Sprintf("%d,%d,%d", a.PadD, a.PadH, a.PadW)
        fmtStr = fmt.Sprintf("%d,%d,%d", a.StrD, a.StrH, a.StrW)
        fmtDil = fmt.Sprintf("%d,%d,%d", a.DilD, a.DilH, a.DilW)
    }

    return fmt.Sprintf(
        "Deconvolution Layer '%s': %s inputs, %s outputs, %s padding, %s stride, %s dilation",
            n.Name(), fmtShape(n.inShape), fmtShape(n.outShape), fmtPad, fmtStr, fmtDil)
}

func(n *Deconvolution) Configure(inObj InputObject) {
    inObj.AssertSingle()
    n.ParameterLayer.Configure(inObj)
    if n.nglayer == nil {
        be := backends.Be()
        a := &n.deconvParams
        inShape := n.inShape
        if len(inShape) == 3 {
            a.K = inShape[0]
            a.P = inShape[1]
            a.Q = inShape[2]
        } else {
            a.K = inShape[0]
            a.M = inShape[1]
            a.P = inShape[2]
            a.Q = inShape[3]
        }
        a.N = be.Bsz()
        n.nglayer = be.NewDeconvLayer(be.DefaultDtype(), a)
        dimI := n.nglayer.DimI()
        c := dimI[0]
        d := dimI[1]
        h := dimI[2]
        w := dimI[3]
        if d == 1 {
            n.outShape = []int{c, h, w}
        } else {
            n.outShape = []int{c, d, h, w}
        }
    }
    if n.weightShape == nil {
        n.weightShape = n.nglayer.DimF2() // (C * R * S, K)
    }
    if n.bsum {
        n.batchSumShape = []int{n.nglayer.C(), 1}
    }
}

func(n *Deconvolution) Fprop(
        inputs []backends.Tensor, inference bool, beta float64) []backends.Tensor {
    assertSingle(inputs)
    be := backends.Be()
    n.inputs = inputs
    be.BpropConv(
        n.nglayer,     // layer
        n.w,           // f
        inputs[0],     // e
        n.outputs[0],  // gradI
        nil,           // x
        nil,           // bias
        n.batchSum,    // bsum
        1.0,           // alpha
        0.0,           // beta
        false,         // relu
        false,         // brelu
        0.0)           // slope
    return n.outputs
}

func(n *Deconvolution) Bprop(
        errors []backends.Tensor, alpha float64, beta float64) []backends.Tensor {
    assertSingle(errors)
    be := backends.Be()
    if n.deltas != nil {
        be.FpropConv(
            n.nglayer,   // layer
            errors[0],   // i
            n.w,         // f
            n.deltas[0], // o
            nil,         // x
            nil,         // bias
            nil,         // bsum
            alpha,       // alpha
            beta,        // beta
            false,       // relu
            false,       // brelu
            0.0)         // slope
    }
    be.UpdateConv(
        n.nglayer,   // layer
        errors[0],   // i
        n.inputs[0], // e
        n.dw,        // gradF
        1.0,         // alpha
        0.0,         // beta
        nil)         // gradBias
    return n.deltas
}

//
//    Linear
//

type Linear struct {
    ParameterLayer
    nout int
    bsum bool
    nin int
    nsteps int
}

var linearInitArgMap = base.ArgMap{
    "nout": base.NewIntArg(),
    "init": base.NewAnyArg(),                     // passthru
    "bsum": base.NewBoolArgOpt(false),
    "name": base.NewAnyArgOpt(""),                // passthru
}

func NewLinear(args ...interface{}) *Linear {
    n := new(Linear)
    n.Init(n, base.MakeArgs(args))
    return n
}

func(n *Linear) Init(self base.Object, args base.Args) {
    args = linearInitArgMap.Expand(args)
    n.ParameterLayer.Init(self, args.Filter([]string{"init", "name"}))
    n.nout = args["nout"].(int)
    n.inputs = nil
    n.bsum = args["bsum"].(bool)
}

func(n *Linear) ClassName() string {
    return "arhat.layers.Linear"
}

func(n *Linear) String() string {
    return fmt.Sprintf("Linear Layer '%s': %d inputs, %d outputs", n.Name(), n.nin, n.nout)
}

func(n *Linear) Configure(inObj InputObject) {
    inObj.AssertSingle()
    n.ParameterLayer.Configure(inObj)
    inShape := interpretInShape(n.inShape)
    n.nin = inShape[0]
    n.nsteps = inShape[1]
    n.outShape = []int{n.nout, n.nsteps}
    if n.weightShape == nil {
        n.weightShape = []int{n.nout, n.nin}
    }
    if n.bsum {
        n.batchSumShape = []int{n.nout, 1}
    }
}

func(n *Linear) Fprop(
        inputs []backends.Tensor, inference bool, beta float64) []backends.Tensor {
    assertSingle(inputs)
    be := backends.Be()
    slice := backends.MakeSlice
    n.inputs = inputs
    if n.actualBsz == base.IntNone && n.actualSeqLen == base.IntNone {
        be.CompoundDot(n.w, n.inputs[0], n.outputs[0], 1.0, beta, false, n.batchSum)
    } else {
        bsz := n.actualBsz
        if bsz == base.IntNone {
            bsz = be.Bsz()
        }
        steps := n.actualSeqLen
        if steps == base.IntNone {
            steps = n.nsteps
        }
        be.CompoundDot(
            n.w, 
            n.inputs[0].GetItem(slice(nil, []int{0, bsz*steps})), 
            n.outputs[0].GetItem(slice(nil, []int{0, bsz*steps})), 
            1.0, 
            beta, 
            false, 
            n.batchSum)
    }
    return n.outputs
}

// SKIPPED: @Layer.accumulates decorator in original code
//     This is 'accumulates' wrapper that manages accParam objects for layers
//     that use them - required for GAN networks only, not relevant at this stage
func(n *Linear) Bprop(
        errors []backends.Tensor, alpha float64, beta float64) []backends.Tensor {
    assertSingle(errors)
    be := backends.Be()
    if n.deltas != nil {
        be.CompoundDot(n.w.T(), errors[0], n.deltas[0], alpha, beta, false, nil)
    }
    be.CompoundDot(errors[0], n.inputs[0].T(), n.dw, 1.0, 0.0, false, nil)
    return n.deltas
}

//
//    SKIPPED: BinaryLinear
//

//
//    Bias
//

type Bias struct {
    ParameterLayer
    y backends.Tensor
    biasSize  int
}

var biasInitArgMap = base.ArgMap{
    "init": base.NewAnyArg(),      // passthru
    "name": base.NewAnyArgOpt(""), // passthru
}

func NewBias(args ...interface{}) *Bias {
    n := new(Bias)
    n.Init(n, base.MakeArgs(args))
    return n
}

func(n *Bias) Init(self base.Object, args base.Args) {
    args = biasInitArgMap.Expand(args)
    n.ParameterLayer.Init(self, args.Filter([]string{"init", "name"}))
    n.y = nil
    n.ownsOutput = false
    n.ownsDelta = false
}

func(n *Bias) ClassName() string {
    return "arhat.layers.Bias"
}

func(n *Bias) String() string {
    if len(n.inShape) == 3 {
        return fmt.Sprintf("Bias Layer '%s': size %d x (%dx%d)",
            n.Name(), n.inShape[0], n.inShape[1], n.inShape[2])
    } else {
        return fmt.Sprintf("Bias Layer '%s': size %d", n.Name(), n.biasSize)
    }
}

func(n *Bias) Configure(inObj InputObject) {
    n.ParameterLayer.Configure(inObj)
    n.outShape = n.inShape
    n.biasSize = n.inShape[0]
    if n.weightShape == nil {
        n.weightShape = []int{n.biasSize, 1}
    }
}

func(n *Bias) Fprop(
        inputs []backends.Tensor, inference bool, beta float64) []backends.Tensor {
    assertSingle(inputs)
    n.inputs = inputs
    n.outputs = inputs
    if n.y == nil || n.y.Base() != n.outputs[0] {
        n.y = n.outputs[0].Reshape([]int{n.biasSize, -1})
    }
    // n.y[] = n.y + n.w
    n.y.Assign(n.y.Add(n.w))
    return n.outputs
}

func(n *Bias) Bprop(
        errors []backends.Tensor, alpha float64, beta float64) []backends.Tensor {
    assertSingle(errors)
    be := backends.Be()
    if n.deltas == nil {
        delta := errors[0].Reshape(n.y.Shape())
        n.deltas = []backends.Tensor{delta}
    }
    // n.dw = sum(n.deltas, axis=1)
    n.dw.Assign(be.Sum(n.deltas[0], 1))
    return errors
}

//
//    Activation
//

type Activation struct {
    LayerBase
    transform transforms.Transform
    nout int
    nglayer backends.Layer
}

var activationInitArgMap = base.ArgMap{
    "transform": transforms.NewTransformArg(),
    "name": base.NewAnyArgOpt(""), // passthru
}

func NewActivation(args ...interface{}) *Activation {
    n := new(Activation)
    n.Init(n, base.MakeArgs(args))
    return n
}

func(n *Activation) Init(self base.Object, args base.Args) {
    args = activationInitArgMap.Expand(args)
    n.LayerBase.Init(self, args.Filter([]string{"name"}))
    n.transform = transforms.ToTransform(args["transform"])
    n.ownsOutput = false
    n.ownsDelta = true
}

func(n *Activation) ClassName() string {
    return "arhat.layers.Activation"
}

func(n *Activation) String() string {
    return fmt.Sprintf("Activation Layer '%s': %s", n.Name(), n.transform.ShortClassName())
}

func(n *Activation) Configure(inObj InputObject) {
    inObj.AssertSingle()
    be := backends.Be()
    n.LayerBase.Configure(inObj)
    n.nglayer = be.NewReluLayer()
    n.outShape = n.inShape
    inShape := interpretInShape(n.inShape)
    n.nout = inShape[0]
}

func(n *Activation) Fprop(
        inputs []backends.Tensor, inference bool, beta float64) []backends.Tensor {
    assertSingle(inputs)
    be := backends.Be()
    n.inputs = inputs
    n.outputs = inputs
    // ACHTUNG: Could relu be set inside FpropTransform?
    _, relu := n.transform.(*transforms.Rectlin)
    be.FpropTransform(
        n.nglayer, 
        n.transform.(backends.Transform), 
        n.inputs[0], 
        n.outputs[0], 
        relu)
    return n.outputs
}

func(n *Activation) Bprop(
        errors []backends.Tensor, alpha float64, beta float64) []backends.Tensor {
    assertSingle(errors)
    be := backends.Be()
    // ACHTUNG: Could relu be set inside BpropTransform?
    _, relu := n.transform.(*transforms.Rectlin)
    be.BpropTransform(
        n.nglayer, 
        n.transform.(backends.Transform), 
        n.outputs[0], 
        errors[0], 
        n.deltas[0], 
        relu)
    return n.deltas
}

//
//    Reshape
//

type Reshape struct {
    LayerBase
    reshape []int
    inShapeT []int
    outShapeT []int
}

var reshapeInitArgMap = base.ArgMap{
    "reshape": base.NewIntListArg(),
    "name": base.NewAnyArgOpt(""), // passthru
}

func NewReshape(args ...interface{}) *Reshape {
    n := new(Reshape)
    n.Init(n, base.MakeArgs(args))
    return n
}

func(n *Reshape) Init(self base.Object, args base.Args) {
    args = reshapeInitArgMap.Expand(args)
    n.LayerBase.Init(self, args.Filter([]string{"name"}))
    n.reshape = base.ToIntList(args["reshape"])
    n.ownsOutput = false
}

func(n *Reshape) ClassName() string {
    return "arhat.layers.Reshape"
}

func(n *Reshape) String() string {
    fmtShape := func(x []int) string {
        s := ""
        for i, v := range x {
            if i != 0 {
                s += ", "
            }
            s += fmt.Sprintf("%d", v)
        }
        return s
    }
    return fmt.Sprintf("Reshape Layer '%s' input shape %s to %s", 
        n.Name(), fmtShape(n.inShape), fmtShape(n.reshape))
}

func(n *Reshape) Configure(inObj InputObject) {
    inObj.AssertSingle()
    bsz := backends.Be().Bsz()

    n.LayerBase.Configure(inObj)
    if len(n.inShape) == 2 {
        n.inShapeT = []int{n.inShape[0], n.inShape[1] * bsz}
    } else {
        n.inShapeT = []int{base.IntsProd(n.inShape), bsz}
    }

    n.outShape = base.IntsCopy(n.reshape)

    dimToKeep := base.IntsFind(n.reshape, 0)
    if dimToKeep >= 0 {
        n.outShape[dimToKeep] = n.inShape[dimToKeep]
    }

    minusDim := base.IntsFind(n.reshape, -1)
    if minusDim >= 0 {
        missingDim := (-base.IntsProd(n.inShape)) / base.IntsProd(n.outShape)
        n.outShape[minusDim] = missingDim
    }

    if len(n.outShape) == 2 {
        n.outShapeT = []int{n.outShape[0], n.outShape[1] * bsz}
    } else {
        n.outShapeT = []int{base.IntsProd(n.outShape), bsz}
    }

    base.Assert(base.IntsProd(n.outShape) == base.IntsProd(n.inShape))
}

func(n *Reshape) Fprop(
        inputs []backends.Tensor, inference bool, beta float64) []backends.Tensor {
    assertSingle(inputs)
    if n.inputs == nil {
        n.inputs = []backends.Tensor{nil}
        n.outputs = []backends.Tensor{nil}
    }
    if !inputs[0].IsContiguous() {
        if n.inputs[0] == nil {
            be := backends.Be()
            n.inputs[0] = be.NewTensor(inputs[0].Shape(), inputs[0].Dtype())
            n.outputs[0] = n.inputs[0].Reshape(n.outShapeT)
        }
        n.inputs[0].Assign(inputs[0])
    } else {
        if n.inputs[0] == nil || n.inputs[0] != inputs[0] {
            n.inputs[0] = inputs[0]
            n.outputs[0] = n.inputs[0].Reshape(n.outShapeT)
        }
    }
    return n.outputs
}

func(n *Reshape) Bprop(
        errors []backends.Tensor, alpha float64, beta float64) []backends.Tensor {
    assertSingle(errors)
    if n.deltas == nil {
        delta := errors[0].Reshape(n.inShapeT)
        n.deltas = []backends.Tensor{delta}
    }
    return n.deltas
}

//
//    DataTransform
//

type DataTransform struct {
    LayerBase
    transform transforms.Transform
    nOut int
}

var dataTransformInitArgMap = base.ArgMap{
    "transform": transforms.NewTransformArg(),
    "name": base.NewAnyArgOpt(""), // passthru
}

func NewDataTransform(args ...interface{}) *DataTransform {
    n := new(DataTransform)
    n.Init(n, base.MakeArgs(args))
    return n
}

func(n *DataTransform) Init(self base.Object, args base.Args) {
    args = dataTransformInitArgMap.Expand(args)
    n.LayerBase.Init(self, args.Filter([]string{"name"}))
    n.transform = transforms.ToTransform(args["transform"])
    n.ownsOutput = false
}

func(n *DataTransform) ClassName() string {
    return "arhat.layers.DataTransform"
}

func(n *DataTransform) String() string {
    return fmt.Sprintf("DataTransform Layer '%s': %s", n.Name(), n.transform.ShortClassName())
}

func(n *DataTransform) Configure(inObj InputObject) {
    n.LayerBase.Configure(inObj)
    n.outShape = n.inShape
    n.nOut = interpretInShape(n.inShape)[0]
}

func(n *DataTransform) Fprop(
        inputs []backends.Tensor, inference bool, beta float64) []backends.Tensor {
    assertSingle(inputs)
    n.inputs = inputs
    n.outputs = inputs
    n.outputs[0].Assign(n.transform.Call(n.inputs[0]))
    return n.outputs
}

func(n *DataTransform) Bprop(
        errors []backends.Tensor, alpha float64, beta float64) []backends.Tensor {
    // Nothing to do
    return nil
}

//
//    CompoundLayer
//

type CompoundLayer struct {
    list []Layer
    activation transforms.Transform
    batchNorm bool
    bias initializers.Initializer
    baseName string
}

var compoundLayerInitArgMap = base.ArgMap{
    "bias": initializers.NewInitializerArgOpt(nil),
    "batch_norm": base.NewBoolArgOpt(false),
    "activation": transforms.NewTransformArgOpt(nil),
    "name": base.NewStringArgOpt(""),
}

func(n *CompoundLayer) Init(args base.Args) {
    args = compoundLayerInitArgMap.Expand(args)
    n.list = nil
    batchNorm := args["batch_norm"].(bool)
    bias := args["bias"]
    if batchNorm && bias != nil {
        base.ArgumentError("Batchnorm and bias cannot be combined")
    }
    n.activation = transforms.ToTransform(args["activation"])
    n.batchNorm = batchNorm
    n.bias = initializers.ToInitializer(bias)
    n.baseName = args["name"].(string)
}

func(n *CompoundLayer) Append(layer Layer) {
    n.list = append(n.list, layer)
}

func(n *CompoundLayer) InitBaseName() {
    if n.baseName == "" {
        n.baseName = n.list[len(n.list)-1].Name()
    }
}

func(n *CompoundLayer) AddPostfilterLayers() {
    n.InitBaseName()
    if n.bias != nil {
        name := n.baseName + "_bias"
        n.Append(NewBias("init", n.bias, "name", name))
    }
    if n.batchNorm {
        name := n.baseName + "_bnorm"
        n.Append(NewBatchNorm("name", name))
    }
    if n.activation != nil {
        name := n.baseName + "_" + n.activation.ShortClassName()
        n.Append(NewActivation("transform", n.activation, "name", name))
    }
}

func(n *CompoundLayer) Layers() []Layer {
    return n.list
}

func(n *CompoundLayer) ToCompoundLayer() *CompoundLayer {
    return n
}

// LayerItem methods

func(n *CompoundLayer) Atom() Layer { 
    return nil
}

func(n *CompoundLayer) List() []Layer {
    return n.list
}

//
//    Affine
//

type Affine struct {
    CompoundLayer
}

var affineInitArgMap = base.ArgMap{
    "nout": base.NewAnyArg(),                     // passthru
    "init": base.NewAnyArg(),                     // passthru
    "bias": base.NewAnyArgOpt(nil),               // passthru
    "batch_norm": base.NewBoolArgOpt(false),      // used + passthru
    "activation": base.NewAnyArgOpt(nil),         // passthru
    "name": base.NewAnyArgOpt(""),                // passthru
}

func NewAffine(args ...interface{}) *Affine {
    n := new(Affine)
    n.Init(base.MakeArgs(args))
    return n
}

func(n *Affine) Init(args base.Args) {
    args = affineInitArgMap.Expand(args)
    n.CompoundLayer.Init(args.Filter([]string{"bias", "batch_norm", "activation", "name"}))
    targs := args.Filter([]string{"nout", "init", "name"})
    targs["bsum"] = args["batch_norm"]
    n.Append(NewLinear(targs))
    n.AddPostfilterLayers()
}

//
//    SKIPPED: BinaryAffine
//

//
//    Conv
//

type Conv struct {
    CompoundLayer
}

var convInitArgMap = base.ArgMap{
    "fshape": base.NewAnyArg(),             // passthru
    "init": base.NewAnyArg(),               // passthru
    "strides": base.NewAnyArgOpt(nil),      // passthru
    "padding": base.NewAnyArgOpt(nil),      // passthru
    "dilation": base.NewAnyArgOpt(nil),     // passthru
    "bias": base.NewAnyArgOpt(nil),         // passthru
    "batch_norm": base.NewAnyArgOpt(false), // passthru
    "activation": base.NewAnyArgOpt(nil),   // passthru
    "name": base.NewAnyArgOpt(""),          // passthru
}

func NewConv(args ...interface{}) *Conv {
    n := new(Conv)
    n.Init(base.MakeArgs(args))
    return n
}

func(n *Conv) Init(args base.Args) {
    args = convInitArgMap.Expand(args)
    n.CompoundLayer.Init(args.Filter([]string{"bias", "batch_norm", "activation", "name"}))
    targs := args.Filter([]string{
        "fshape",
        "strides",
        "padding",
        "dilation",
        "init",
        "name",
    })
    targs["bsum"] = args["batch_norm"]
    n.Append(NewConvolution(targs))
    n.AddPostfilterLayers()
}

//
//    Deconv
//

type Deconv struct {
    CompoundLayer
}

var deconvInitArgMap = base.ArgMap{
    "fshape": base.NewAnyArg(),             // passthru
    "init": base.NewAnyArg(),               // passthru
    "strides": base.NewAnyArgOpt(nil),      // passthru
    "padding": base.NewAnyArgOpt(nil),      // passthru
    "dilation": base.NewAnyArgOpt(nil),     // passthru
    "bias": base.NewAnyArgOpt(nil),         // passthru
    "batch_norm": base.NewAnyArgOpt(false), // passthru
    "activation": base.NewAnyArgOpt(nil),   // passthru
    "name": base.NewAnyArgOpt(""),          // passthru
}

func NewDeconv(args ...interface{}) *Deconv {
    n := new(Deconv)
    n.Init(base.MakeArgs(args))
    return n
}

func(n *Deconv) Init(args base.Args) {
    args = deconvInitArgMap.Expand(args)
    n.CompoundLayer.Init(args.Filter([]string{"bias", "batch_norm", "activation", "name"}))
    targs := args.Filter([]string{
        "fshape",
        "strides",
        "padding",
        "dilation",
        "init",
    })
    targs["bsum"] = args["batch_norm"]
    n.Append(NewDeconvolution(targs))
    n.AddPostfilterLayers()
}

//
//    LRN
//

type LRN struct {
    LayerBase
    j int
    depth int
    alpha float64
    beta float64
    ascale float64
    bpower float64
    lrnParams backends.LrnParams
    nglayer backends.LrnLayer
    denom backends.Tensor
}

var lrnInitArgMap = base.ArgMap{
    "depth": base.NewIntArg(),
    "alpha": base.NewFloatArgOpt(1.0),
    "beta": base.NewFloatArgOpt(0.0),
    "ascale": base.NewFloatArgOpt(1.0),
    "bpower": base.NewFloatArgOpt(1.0),
    "name": base.NewAnyArgOpt(""), // passthru
}

func NewLRN(args ...interface{}) *LRN {
    n := new(LRN)
    n.Init(n, base.MakeArgs(args))
    return n
}

func(n *LRN) Init(self base.Object, args base.Args) {
    args = lrnInitArgMap.Expand(args)
    n.LayerBase.Init(self, args.Filter([]string{"name"}))
    depth := args["depth"].(int)
    n.j = depth
    n.depth = depth
    n.alpha = args["alpha"].(float64)
    n.beta = args["beta"].(float64)
    n.ascale = args["ascale"].(float64)
    n.bpower = args["bpower"].(float64)
    n.ownsDelta = true
    n.lrnParams.Init()
    n.nglayer = nil
}

func(n *LRN) ClassName() string {
    return "arhat.layers.LRN"
}

func(n *LRN) Configure(inObj InputObject) {
    inObj.AssertSingle()
    n.LayerBase.Configure(inObj)
    if n.nglayer == nil {
        be := backends.Be()
        a := &n.lrnParams
        inShape := n.inShape
        if len(n.inShape) == 3 {
            a.C = inShape[0]
            a.H = inShape[1]
            a.W = inShape[2]
        } else {
            a.C = inShape[0]
            a.D = inShape[1]
            a.H = inShape[2]
            a.W = inShape[3]
        }
        a.N = be.Bsz()
        n.nglayer = be.NewLrnLayer(be.DefaultDtype(), a)
        n.outShape = n.inShape
    }
}

func(n *LRN) Allocate(sharedOutputs backends.Tensor, accumulateUpdates bool) {
    be := backends.Be()
    n.LayerBase.Allocate(sharedOutputs, false)
    n.denom = be.Iobuf(n.inShape, nil, base.DtypeNone, "", true, nil)
}

func(n *LRN) Fprop(
        inputs []backends.Tensor, inference bool, beta float64) []backends.Tensor {
    assertSingle(inputs)
    be := backends.Be()
    n.inputs = inputs
    be.FpropLrn(
        n.nglayer,
        inputs[0], 
        n.outputs[0], 
        n.denom,
        n.alpha, 
        n.beta, 
        n.ascale, 
        n.bpower)
    return n.outputs
}

func(n *LRN) Bprop(
        errors []backends.Tensor, alpha float64, beta float64) []backends.Tensor {
    assertSingle(errors)
    be := backends.Be()
    if n.deltas != nil {
        be.BpropLrn(
            n.nglayer,
            n.inputs[0], 
            n.outputs[0], 
            errors[0], 
            n.deltas[0], 
            n.denom,
            n.alpha, 
            n.beta, 
            n.ascale, 
            n.bpower)
    }
    return n.deltas
}

//
//    Dropout
//

type Dropout struct {
    LayerBase
    keep float64
    keepMask backends.Tensor
    caffeMode bool
    trainScaling float64
    nOut int
}

var dropoutInitArgMap = base.ArgMap{
    "keep": base.NewFloatArgOpt(0.5),
    "name": base.NewAnyArgOpt(""), // passthru
}

func NewDropout(args ...interface{}) *Dropout {
    n := new(Dropout)
    n.Init(n, base.MakeArgs(args))
    return n
}

func(n *Dropout) Init(self base.Object, args base.Args) {
    args = dropoutInitArgMap.Expand(args)
    n.LayerBase.Init(self, args.Filter([]string{"name"}))
    keep := args["keep"].(float64)
    n.keep = keep
    n.keepMask = nil
    n.caffeMode = backends.Be().CheckCaffeCompat()
    if n.caffeMode {
        // scaling factor during training
        n.trainScaling = 1.0 / keep
    } else {
        // override scaling factor to retain binary mask
        n.trainScaling = 1.0 
    }
    n.ownsOutput = false
}

func(n *Dropout) ClassName() string {
    return "arhat.layers.Dropout"
}

func(n *Dropout) String() string {
    return fmt.Sprintf("Dropout Layer '%s': %d inputs and outputs, keep %d%% (caffe_compat %t)",
        n.Name(), n.nOut, int(100.0*n.keep), n.caffeMode)
}

func(n *Dropout) Configure(inObj InputObject) {
    inObj.AssertSingle()
    n.LayerBase.Configure(inObj)
    n.outShape = n.inShape
    n.nOut = interpretInShape(n.inShape)[0]
}

func(n *Dropout) Allocate(sharedOutputs backends.Tensor, accumulateUpdates bool) {
    be := backends.Be()
    n.LayerBase.Allocate(sharedOutputs, false)
    n.keepMask = be.Iobuf(n.outShape, nil, base.DtypeNone, "", true, nil)
}

func(n *Dropout) Fprop(
        inputs []backends.Tensor, inference bool, beta float64) []backends.Tensor {
    assertSingle(inputs)
    be := backends.Be()

    n.inputs = inputs
    n.outputs = inputs
    if inference {
        return n.fpropInference(inputs)
    }

    be.MakeBinaryMask(n.keepMask, n.keep)
    // n.outputs[0][] = n.keepMask * inputs[0] * n.transScaling
    n.outputs[0].Assign(n.keepMask.Mul(inputs[0]).Mul(be.Float(n.trainScaling)))

    return n.outputs
}

func(n *Dropout) fpropInference(inputs []backends.Tensor) []backends.Tensor {
    if !n.caffeMode {
        be := backends.Be()
        // n.outputs[0][] = inputs[0] * n.keep
        n.outputs[0].Assign(inputs[0].Mul(be.Float(n.keep)))
    }
    return n.outputs
}

func(n *Dropout) Bprop(
        errors []backends.Tensor, alpha float64, beta float64) []backends.Tensor {
    assertSingle(errors)
    be := backends.Be()
    if n.deltas == nil {
        n.deltas = errors
    }
    // n.deltas[] = n.keepMask * errors * alpha * n.trainScaling + beta * errors
    fAlpha := be.Float(alpha)
    fBeta := be.Float(beta)
    trainScaling := be.Float(n.trainScaling)
    n.deltas[0].Assign(
        n.keepMask.Mul(errors[0]).Mul(fAlpha).Mul(trainScaling).Add(fBeta.Mul(errors[0])))
    return n.deltas
}

//
//    SKIPPED: LookupTable
//

//
//    Cost
//

type Cost interface {
    base.Object
    CostFunc() transforms.Cost
    Outputs() []backends.Tensor
    Deltas() []backends.Tensor
    Initialize(inObj Layer) 
    GetCost(inputs []backends.Tensor, targets []backends.Tensor) backends.Value
    GetErrors(inputs []backends.Tensor, targets []backends.Tensor) []backends.Tensor
}

//
//    GeneralizedCost
//

type GeneralizedCost struct {
    base.ObjectBase
    costFunc transforms.Cost
    outputs []backends.Tensor
    deltas []backends.Tensor
    costBuffer backends.Tensor
    prevLayer Layer
    nstep int
}

var generalizedCostInitArgMap = base.ArgMap{
    "costfunc": transforms.NewCostArg(),
    "name": base.NewAnyArgOpt(""), // passthru
}

func NewGeneralizedCost(args ...interface{}) *GeneralizedCost {
    n := new(GeneralizedCost)
    n.Init(n, base.MakeArgs(args))
    return n
}

func(n *GeneralizedCost) Init(self base.Object, args base.Args) {
    args = generalizedCostInitArgMap.Expand(args)
    n.ObjectBase.Init(self, args.Filter([]string{"name"}))
    n.costFunc = transforms.ToCost(args["costfunc"])
    n.outputs = nil
    n.deltas = nil
    be := backends.Be()
    n.costBuffer = be.NewTensor([]int{1, 1}, base.DtypeNone)
}

func(n *GeneralizedCost) ClassName() string {
    return "arhat.layers.GeneralizedCost"
}

func(n *GeneralizedCost) CostFunc() transforms.Cost {
    return n.costFunc
}

func(n *GeneralizedCost) Outputs() []backends.Tensor {
    return n.outputs
}

func(n *GeneralizedCost) Deltas() []backends.Tensor {
    return n.deltas
}

func(n *GeneralizedCost) Initialize(inObj Layer) {
    be := backends.Be()
    n.prevLayer = inObj
    inShape := interpretInShape(inObj.OutShape())
    n.nstep = inShape[1]
    output := be.Iobuf([]int{1, n.nstep}, nil, base.DtypeNone, "", false, nil)
    n.outputs = []backends.Tensor{output}
    delta := be.Iobuf(inObj.OutShape(), nil, base.DtypeNone, "", false, nil)
    n.deltas = []backends.Tensor{delta}
}

func(n *GeneralizedCost) GetCost(
        inputs []backends.Tensor, targets []backends.Tensor) backends.Value {
    assertSingle(inputs)
    assertSingle(targets)
    be := backends.Be()
    n.outputs[0].Assign(n.costFunc.Call(inputs[0], targets[0]))
    n.costBuffer.Assign(be.Mean(n.outputs[0], 1))
    return n.costBuffer.GetScalar()
}

func(n *GeneralizedCost) GetErrors(
        inputs[] backends.Tensor, targets []backends.Tensor) []backends.Tensor {
    assertSingle(inputs)
    assertSingle(targets)
    n.deltas[0].Assign(n.costFunc.Bprop(inputs[0], targets[0]))
    return n.deltas
}

//
//    SKIPPED: GeneralizedGANCost
//

//
//    GeneralizedCostMask
//

type GeneralizedCostMask struct {
    GeneralizedCost
    weights float64 // TODO: Shall this be Tensor?
}

var generalizedCostMaskInitArgMap = base.ArgMap{
    "costfunc": transforms.NewCostArg(),
    "weights": base.NewFloatArgOpt(1.0),
    "name": base.NewAnyArgOpt(""), // passthru
}

func NewGeneralizedCostMask(args ...interface{}) *GeneralizedCostMask {
    n := new(GeneralizedCostMask)
    n.Init(n, base.MakeArgs(args))
    return n
}

func(n *GeneralizedCostMask) Init(self base.Object, args base.Args) {
    args = generalizedCostMaskInitArgMap.Expand(args)
    n.GeneralizedCost.Init(self, args.Filter([]string{"costfunc", "name"}))
    n.weights = args["weights"].(float64)
}

func(n *GeneralizedCostMask) ClassName() string {
    return "arhat.layers.GeneralizedCostMask"
}

func(n *GeneralizedCostMask) GetCost(
        inputs []backends.Tensor, targets []backends.Tensor) backends.Value {
    assertSingle(inputs)
    base.Assert(len(targets) == 2) // [0] targets [1] masks
    be := backends.Be()
    mask := targets[1]
    maskedInput := inputs[0].Mul(mask)
    maskedTargets := targets[0].Mul(mask)
    n.outputs[0].Assign(n.costFunc.Call(maskedInput, maskedTargets))
    n.costBuffer.Assign(be.Mean(n.outputs[0], 1).Mul(be.Float(n.weights)))
    return n.costBuffer.GetScalar()
}

func(n *GeneralizedCostMask) GetErrors(
        inputs []backends.Tensor, targets []backends.Tensor) []backends.Tensor {
    assertSingle(inputs)
    base.Assert(len(targets) == 2) // [0] targets [1] masks
    be := backends.Be()
    mask := targets[1]
    n.deltas[0].Assign(n.costFunc.Bprop(inputs[0], targets[0]).Mul(mask).Mul(be.Float(n.weights)))
    return n.deltas 
}

//
//    BatchNorm
//

type BatchNorm struct {
    LayerBase
    allparams []backends.Tensor
    errorView backends.Tensor
    rho float64
    eps float64
    states [2]*States
    relu bool
    beta backends.Tensor
    gamma backends.Tensor
    gmean backends.Tensor
    gvar backends.Tensor
    statsDtype base.Dtype
    binary bool
    nin int
    nsteps int
    nfm int
    nglayer backends.BatchNormLayer
    y backends.Tensor
    xvar backends.Tensor
    xsum backends.Tensor
    accumulateUpdates bool
    computeBatchSum bool
    params []backends.Tensor
    gradBeta backends.Tensor
    gradGamma backends.Tensor
    gradParams []backends.Tensor
    infParams []backends.Tensor
    accGradBeta backends.Tensor
    accGradGamma backends.Tensor
    accParams []accParam
}

var batchNormInitArgMap = base.ArgMap{
    "rho": base.NewFloatArgOpt(0.9),
    "eps": base.NewFloatArgOpt(1e-3),
    "name": base.NewAnyArgOpt(""),       // passthru
    "binary": base.NewBoolArgOpt(false),
}

func NewBatchNorm(args ...interface{}) *BatchNorm {
    n := new(BatchNorm)
    n.Init(n, base.MakeArgs(args))
    return n
}

func(n *BatchNorm) Init(self base.Object, args base.Args) {
    args = batchNormInitArgMap.Expand(args)
    n.LayerBase.Init(self, args.Filter([]string{"name"}))
    n.allparams = nil
    n.hasParams = true
    n.ownsDelta = true
    n.errorView = nil
    n.rho = args["rho"].(float64)
    n.eps = args["eps"].(float64)
    n.states[0] = NewStates()
    n.states[1] = NewStates()
    n.relu = false
    n.beta = nil
    n.gamma = nil
    n.gmean = nil
    n.gvar = nil
    if backends.Be().DefaultDtype() == base.Float64 {
        n.statsDtype = base.Float64
    } else {
        n.statsDtype = base.Float32
    }
    n.binary = args["binary"].(bool)
}

func(n *BatchNorm) ClassName() string {
    return "arhat.layers.BatchNorm"
}

func(n *BatchNorm) Configure(inObj InputObject) {
    inObj.AssertSingle()
    n.LayerBase.Configure(inObj)
    n.outShape = n.inShape
    inShape := interpretInShape(n.inShape)
    n.nin = inShape[0]
    n.nsteps = inShape[1]
    n.nfm = n.inShape[0]
    n.nglayer = backends.Be().NewBatchNormLayer(n.inShape)
}

func(n *BatchNorm) Allocate(sharedOutputs backends.Tensor, accumulateUpdates bool) {
    be := backends.Be()
    n.LayerBase.Allocate(sharedOutputs, false)
    n.y = n.outputs[0].Reshape([]int{n.nfm, -1})
    n.xvar = be.NewTensor([]int{n.nfm, 1}, n.statsDtype)
    n.xvar.Fill(0)
    n.accumulateUpdates = accumulateUpdates
    if n.allparams == nil {
        n.InitParams(n.nfm)
    }
    // ACHTUNG: Original code also matches 'prevLayer' with 'true': 
    //     no idea what this is supposed to mean
    if n.prevLayer == nil || n.prevLayer.BatchSum() == nil {
        n.xsum = be.NewTensor([]int{n.nfm, 1}, n.statsDtype)
        n.xsum.Fill(0)
        n.computeBatchSum = true
    } else {
        n.xsum = n.prevLayer.BatchSum()
        n.computeBatchSum = false
    }
}

func(n *BatchNorm) InitParams(dim0 int) {
    be := backends.Be()

    makeParam := func(v int) backends.Tensor {
        result := be.NewTensor([]int{dim0, 1}, n.statsDtype)
        if v != base.IntNone {
            result.Fill(v)
        }
        return result
    }

    n.beta = makeParam(0)
    n.gamma = makeParam(1)
    n.params = []backends.Tensor{n.beta, n.gamma}

    n.gradBeta = makeParam(0)
    n.gradGamma = makeParam(0)
    n.gradParams = []backends.Tensor{n.gradBeta, n.gradGamma}

    n.gmean = makeParam(0)
    n.gvar = makeParam(0)
    n.infParams = []backends.Tensor{n.gmean, n.gvar}

    n.allparams = []backends.Tensor{n.params[0], n.params[1], n.infParams[0], n.infParams[1]}

    // Scratch buffers for accumulation
    if n.accumulateUpdates {
        n.accGradBeta = makeParam(base.IntNone)
        n.accGradGamma = makeParam(base.IntNone)
        n.accParams = []accParam{
            accParam{n.accGradBeta, n.gradBeta}, 
            accParam{n.accGradGamma, n.gradGamma},
        }
    }
}

func(n *BatchNorm) GetParams() []Param {
    return []Param{
        Param{n.params[0], n.gradParams[0], n.states[0]},
        Param{n.params[1], n.gradParams[1], n.states[1]},
    }
}

func(n *BatchNorm) Fprop(
        inputs []backends.Tensor, inference bool, beta float64) []backends.Tensor {
    assertSingle(inputs)
    be := backends.Be()

    if n.inputs == nil {
        n.inputs = make([]backends.Tensor, 1)
    }
    if n.inputs[0] == nil || n.inputs[0].Base() != inputs[0] {
        n.inputs[0] = inputs[0].Reshape([]int{n.nfm, -1})
    }

    be.CompoundFpropBn(
        n.inputs[0],       // x
        n.xsum,            // xsum
        n.xvar,            // xvar
        n.gmean,           // gmean
        n.gvar,            // gvar
        n.gamma,           // gamma
        n.beta,            // beta
        n.y,               // y
        n.eps,             // eps
        n.rho,             // rho
        n.computeBatchSum, // computeBatchSum
        beta,              // accumbeta
        n.relu,            // relu
        n.binary,          // binary
        inference,         // inference
        n.outputs[0],      // outputs
        n.nglayer)         // layer

    return n.outputs
}

// SKIPPED: @Layer.accumulates decorator in original code
//     This is 'accumulates' wrapper that manages accParam objects for layers
//     that use them - required for GAN networks only, not relevant at this stage
func(n *BatchNorm) Bprop(
        errors []backends.Tensor, alpha float64, beta float64) []backends.Tensor {
    assertSingle(errors)
    be := backends.Be()

    base.Assert(alpha == 1.0 && beta == 0.0)
    if n.errorView == nil {
        n.errorView = errors[0].Reshape([]int{n.nfm, -1})
    }

    be.CompoundBpropBn(
        n.deltas[0], // deltaOut
        n.gradGamma, // gradGamma
        n.gradBeta,  // gradBeta
        n.errorView, // deltaIn
        n.inputs[0], // x
        n.xsum,      // xsum
        n.xvar,      // xvar
        n.gamma,     // gamma
        n.eps,       // eps
        n.binary,    // binary
        n.nglayer)   // layer

    return n.deltas
}

func(n *BatchNorm) ReadParams(r ParamReader) {
    r.Read(n.beta)
    r.Read(n.gamma)
    r.Read(n.gmean)
    r.Read(n.gvar)
}

func(n *BatchNorm) WriteParams(w ParamWriter) {
    w.Write(n.beta)
    w.Write(n.gamma)
    w.Write(n.gmean)
    w.Write(n.gvar)
}

//
//    SKIPPED: BatchNormAutodiff
//

//
//    SKIPPED: ShiftBatchNorm
//

//
//    SKIPPED: RoiPooling
//

