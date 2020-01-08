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
    "fragata/arhat/backends"
    "fragata/arhat/base"
    "fragata/arhat/transforms"
    "strings"
)

func toBranchNode(n Layer) *BranchNode {
    if b, ok := n.(*BranchNode); ok {
        return b
    }
    return nil
}

func toSequential(n Layer) *Sequential {
    if s, ok := n.(SequentialTrait); ok {
        return s.ToSequential()
    }
    return nil
}

func flatten(items []LayerItem) []Layer {
    var result []Layer
    for _, item := range items {
        list := item.List()
        if list == nil {
            layer := item.Atom();
            base.Assert(layer != nil)
            result = append(result, layer)
        } else {
            for _, layer := range list {
                result = append(result, layer)
            }
        }
    }
    return result
}

//
//    DeltasTree
//

type DeltasTree struct {
    base.Object
    parent *DeltasTree
    child *DeltasTree
    buffers [2]backends.Tensor
    maxShape int
}

func NewDeltasTree(parent *DeltasTree) *DeltasTree {
    t := new(DeltasTree)
    t.Init(parent)
    return t
}

func(t *DeltasTree) Init(parent *DeltasTree) {
    t.parent = parent
    t.child = nil
    t.maxShape = 0
}

func(t *DeltasTree) Descend() *DeltasTree {
    if t.child == nil {
        t.child = NewDeltasTree(nil)
    }
    return t.child
}

func(t *DeltasTree) Ascend() *DeltasTree {
    return t.parent
}

func(t *DeltasTree) ProcLayer(layer Layer) {
    inSize := backends.Be().SharedIobufSize(layer.InShape())
    if inSize > t.maxShape {
        t.maxShape = inSize
    }
}

func(t *DeltasTree) AllocateBuffers() {
    be := backends.Be()
    if t.child != nil {
        t.child.AllocateBuffers()
    }
    for ind, _ := range t.buffers {
        if t.buffers[ind] == nil && t.maxShape > 0 {
            t.buffers[ind] = be.Iobuf([]int{t.maxShape}, nil, base.DtypeNone, "", false, nil)
        }
    }
}

func(t *DeltasTree) ReverseBuffers() {
    t.buffers[0], t.buffers[1] = t.buffers[1], t.buffers[0]
}

//
//    LayerContainerTrait
//

type LayerContainerTrait interface {
    Layer
    Layers() []Layer
    LayersToOptimize() []Layer
    ToLayerContainer() *LayerContainer
}

//
//    LayerContainer
//

type LayerContainer struct {
    LayerBase
    layers []Layer
}

var layerContainerInitArgMap = base.ArgMap{
    "name": base.NewAnyArgOpt(""), // passthru
}

func(n *LayerContainer) Init(self base.Object, args base.Args) {
    args = layerContainerInitArgMap.Expand(args)
    n.LayerBase.Init(self, args.Filter([]string{"name"}))
}

func(n *LayerContainer) Layers() []Layer {
    return n.layers
}

func(n *LayerContainer) LayersToOptimize() []Layer {
    var lto []Layer
    for _, l := range n.layers {
        if c, ok := l.(LayerContainerTrait); ok {
            p := c.LayersToOptimize()
            for _, q := range p {
                lto = append(lto, q)
            }
        } else if l.HasParams() {
            // TODO: if have l.init && l.init is initializers.Identity { continue }
            lto = append(lto, l)
        }
    }
    return lto
}

func(n *LayerContainer) NestedStr(level int) string {
    padstr := "\n" + strings.Repeat("  ", level)
    ss := strings.Repeat("  ", level) + n.Self().ClassName()
    for _, l := range n.layers {
        ss += padstr + l.NestedStr(level+1)
    }
    return ss
}

func (n *LayerContainer) SetDeltas(deltaBuffers *DeltasTree) {
    for _, l := range n.layers {
        l.SetDeltas(deltaBuffers)
    }
}

func (n *LayerContainer) ToLayerContainer() *LayerContainer {
    return n
}

func (n *LayerContainer) ReadParams(r ParamReader) {
    for _, layer := range n.layers {
        layer.ReadParams(r)
    }
}

func (n *LayerContainer) WriteParams(w ParamWriter) {
    for _, layer := range n.layers {
        layer.WriteParams(w)
    }
}

//
//    SequentialTrait
//

type SequentialTrait interface {
    LayerContainerTrait
    ToSequential() *Sequential
}

//
//    Sequential
//

type Sequential struct {
    LayerContainer
    xlayers []Layer
    accumulateUpdates bool
    globalDeltas *DeltasTree
}

var sequentialInitArgMap = base.ArgMap{
    "name": base.NewAnyArgOpt(""), // passthru
}

func NewSequential(layers []LayerItem, args ...interface{}) *Sequential {
    n := new(Sequential)
    n.Init(n, layers, base.MakeArgs(args))
    return n
}

func(n *Sequential) Init(self base.Object, layers []LayerItem, args base.Args) {
    args = sequentialInitArgMap.Expand(args)
    n.LayerContainer.Init(self, args.Filter([]string{"name"}))
    base.AssertMsg(len(layers) != 0, "Provide layers")
    n.layers = flatten(layers)
    n.xlayers = nil
    for _, x := range n.layers {
        if toBranchNode(x) == nil {
            n.xlayers = append(n.xlayers, x)
        }
    }
    root := n.xlayers[0] // TODO: Check len(xlayers) before?
    rootOk := false
    switch root.(type) {
    case *Dropout, *DataTransform:
        rootOk = true
    default:
        rootOk = root.OwnsOutput()
    }
    base.AssertMsg(rootOk, "Sequential root must own outputs")
}

func(n *Sequential) ClassName() string {
    return "arhat.layers.Sequential"
}

func(n *Sequential) Configure(inObj InputObject) {
    configLayers := n.layers
    if inObj.IsNil() {
        inObj = MakeInputObject(n.layers[0])
        // Remove the initial branch nodes from the layers
        for idx, l := range n.layers {
            if toBranchNode(l) == nil {
                configLayers = n.layers[idx:]
                break
            }
        }
    }

    n.LayerContainer.Configure(inObj)
    var prevLayer Layer
    for _, l := range configLayers {
        l.Configure(inObj)
        inObj = MakeInputObject(l)
        if prevLayer != nil {
            prevLayer.SetNext(l)
        }
        prevLayer = l
    }
    n.outShape = prevLayer.OutShape()
}

func(n *Sequential) Allocate(sharedOutputs backends.Tensor, accumulateUpdates bool) {
    n.accumulateUpdates = accumulateUpdates
    // get the last layer that owns its output
    var lastAllocLayer Layer
    for _, l := range n.layers {
        if l.OwnsOutput() {
            lastAllocLayer = l
        }
    }
    lastAllocLayer.Allocate(sharedOutputs, accumulateUpdates)
    for _, l := range n.layers {
        l.Allocate(nil, accumulateUpdates)
    }
}

func(n *Sequential) AllocateDeltas(globalDeltas *DeltasTree) {
    if globalDeltas == nil {
        n.globalDeltas = NewDeltasTree(nil)

        stInd := 1
        if n.layers[0].NestDeltas() {
            stInd = 0
        }
        for _, layer := range n.layers[stInd:] {
            layer.AllocateDeltas(n.globalDeltas)
        }

        n.globalDeltas.AllocateBuffers()
    } else {
        n.globalDeltas = globalDeltas
    }

    n.SetDeltas(n.globalDeltas)
}

func(n *Sequential) Fprop(
        inputs []backends.Tensor, inference bool, beta float64) []backends.Tensor {
    x := inputs

    last := len(n.layers) - 1
    for i, l := range n.layers {
        // SKIPPED: DistributeData
        // SKIPPED: ConvertData
        if i == last {
            x = l.Fprop(x, inference, beta)
        } else {
            // beta=0.0 is default Fprop argument for all supported layers
            x = l.Fprop(x, inference, 0.0)
        }
    }

    // SKIPPED: RevertTensors
    return x
}

func(n *Sequential) Bprop(
        errors []backends.Tensor, alpha float64, beta float64) []backends.Tensor {
    for i := len(n.xlayers) - 1; i >= 0; i-- {
        l := n.xlayers[i]
        if toBranchNode(l.PrevLayer()) != nil || i == 0 {
            errors = l.Bprop(errors, alpha, beta)
        } else {
            // alpha=1.0, beta=0.0 are default Bprop arguments for all supported layers
            // TODO: Use FloatNone universally as default values instead?
            errors = l.Bprop(errors, 1.0, 0.0)
        }
    }
    return n.xlayers[0].Deltas()
}

func(n *Sequential) GetTerminal() []Layer {
    return n.layers[len(n.layers)-1].GetTerminal()
}

func(n *Sequential) ToSequential() *Sequential {
    return n
}

//
//    SKIPPED: GenerativeAdversarial (GAN)
//

//
//    Tree
//

type Tree struct {
    LayerContainer
    alphas []float64
    betas []float64
    multiOutShape [][]int
}

var treeInitArgMap = base.ArgMap{
    "name": base.NewAnyArgOpt(""), // passthru
    "alphas": base.NewFloatListArgOpt(nil),
}

func NewTree(layers []LayerItem, args ...interface{}) *Tree {
    n := new(Tree)
    n.Init(n, layers, base.MakeArgs(args))
    return n
}

func(n *Tree) Init(self base.Object, layers []LayerItem, args base.Args) {
    args = treeInitArgMap.Expand(args)
    n.LayerContainer.Init(self, args.Filter([]string{"name"}))
    n.layers = nil
    for _, l := range layers {
        if a := l.Atom(); toSequential(a) != nil {
            n.layers = append(n.layers, a)
        } else {
            n.layers = append(n.layers, NewSequential([]LayerItem{l}))
        }
    }
    n.alphas = base.ToFloatList(args["alphas"])
    if n.alphas == nil {
        count := len(n.layers)
        n.alphas = make([]float64, count)
        for i := 0; i < count; i++ {
            n.alphas[i] = 1.0
        }
    }

    // alphas and betas are used for back propagation
    // We want to ensure that the branches are ordered according to the origin of their roots
    // then the betas will be 0 for the last appearance of the root, and 1 for the rest,
    // but the trunk will always be 1 (since it contains all of the branch nodes)
    count := len(n.layers)
    n.betas = make([]float64, count)
    var nextRoot Layer
    for i := count - 1; i >= 0; i-- {
        l := n.layers[i]
        root := toSequential(l).layers[0]
        beta := 0.0
        if root == nextRoot || toBranchNode(root) == nil {
            beta = 1.0
        }
        nextRoot = root
        n.betas[i] = beta
    }
}

func(n *Tree) ClassName() string {
    return "arhat.layers.Tree"
}

func(n *Tree) MultiOutShape() [][]int {
    return n.multiOutShape
}

func(n *Tree) NestedStr(level int) string {
    // TODO: Check indentation (may need extra padding to look nice)
    ss := n.Self().ClassName()
    for _, l := range n.layers {
        ss += "\n" + l.NestedStr(level+1)
    }
    return ss
}

func(n *Tree) Configure(inObj InputObject) {
    inObj.AssertSingle()
    n.LayerContainer.Configure(inObj)
    n.layers[0].Configure(inObj)
    for _, l := range n.layers[1:] {
        l.Configure(MakeInputObject(nil))
    }
    n.multiOutShape = make([][]int, len(n.layers))
    for i, l := range n.layers {
        n.multiOutShape[i] = l.OutShape()
    }
}

func(n *Tree) Allocate(sharedOutputs backends.Tensor, accumulateUpdates bool) {
    for _, l := range n.layers {
        l.Allocate(nil, false)
    }
    n.outputs = make([]backends.Tensor, len(n.layers))
    for i, l := range n.layers {
        outputs := l.Outputs()
        assertSingle(outputs)
        if len(outputs) != 0 {            
            n.outputs[i] = outputs[0]
        }
    }
}

func(n *Tree) AllocateDeltas(globalDeltas *DeltasTree) {
    for i := len(n.layers) - 1; i >= 0; i-- {
        l := n.layers[i]
        l.AllocateDeltas(globalDeltas)
    }
}

func(n *Tree) Fprop(
        inputs []backends.Tensor, inference bool, beta float64) []backends.Tensor {
    count := len(n.layers)
    out := make([]backends.Tensor, count)
    x := n.layers[0].Fprop(inputs, inference, 0.0)
    assertSingle(x)
    out[0] = x[0]
    for i := 1; i < count; i++ {
        l := n.layers[i]
        x = l.Fprop(nil, inference, 0.0)
        assertSingle(x)
        out[i] = x[0]
    }
    return out
}

func(n *Tree) Bprop(
        errors []backends.Tensor, alpha float64, beta float64) []backends.Tensor {
    for i := len(n.layers) - 1; i >= 0; i-- {
        l := n.layers[i]
        e := errors[i:i+1]
        a := n.alphas[i]
        b := n.betas[i]
        l.Bprop(e, a, b)
    }
    // output is never used, discard
    return nil
}

func(n *Tree) GetTerminal() []Layer {
    t := make([]Layer, len(n.layers))
    for i, l := range n.layers {
        // all layers are Sequential; expected to return one terminal
        t[i] = l.GetTerminal()[0]
    }
    return t
}

//
//    SingleOutputTree
//

type SingleOutputTree struct {
    Tree
}

func NewSingleOutputTree(layers []LayerItem, args ...interface{}) *SingleOutputTree {
    n := new(SingleOutputTree)
    n.Init(n, layers, base.MakeArgs(args))
    return n
}

func(n *SingleOutputTree) ClassName() string {
    return "arhat.layers.SingleOutputTree"
}

func(n *SingleOutputTree) Fprop(
        inputs []backends.Tensor, inference bool, beta float64) []backends.Tensor {
    x := n.layers[0].Fprop(inputs, inference, 0.0)
    assertSingle(x)
    if inference {
        return x
    }
    count := len(n.layers)
    out := make([]backends.Tensor, count)
    out[0] = x[0]
    for i := 1; i < count; i++ {
        l := n.layers[i]
        x = l.Fprop(nil, inference, 0.0)
        assertSingle(x)
        out[i] = x[0]
    }
    return out
}

//
//    Broadcast
//

type Broadcast struct {
    LayerContainer
}

var broadcastInitArgMap = base.ArgMap{
    "name": base.NewAnyArgOpt(""), // passthru
}

func(n *Broadcast) Init(self base.Object, layers []LayerItem, args base.Args) {
    args = broadcastInitArgMap.Expand(args)
    n.LayerContainer.Init(self, args.Filter([]string{"name"}))
    n.layers = nil
    for _, l := range layers {
        if a := l.Atom(); toSequential(a) != nil {
            n.layers = append(n.layers, a)
        } else {
            n.layers = append(n.layers, NewSequential([]LayerItem{l}))
        }
    }
    n.ownsOutput = true
    n.outputs = nil
}

func(n *Broadcast) NestDeltas() bool {
    return true
}

func(n *Broadcast) Configure(inObj InputObject) {
    n.LayerContainer.Configure(inObj)

    // Receiving from single source -- distribute to branches
    for _, l := range n.layers {
        l.Configure(inObj)
    }
}

func(n *Broadcast) AllocateDeltas(globalDeltas *DeltasTree) {
    nestedDeltas := globalDeltas.Descend()
    for _, layer := range n.layers {
        for i, sublayer := range toSequential(layer).Layers() {
            if i == 0 {
                sublayer.AllocateDeltas(globalDeltas)
            } else {
                sublayer.AllocateDeltas(nestedDeltas)
            }
        }
    }
}

func (n *Broadcast) SetDeltas(deltaBuffers *DeltasTree) {
    bottomBuffer := deltaBuffers.buffers[0]

    nestedDeltas := deltaBuffers.Descend()
    base.Assert(nestedDeltas != nil)
    for _, l := range n.layers {
        for i, sublayer := range toSequential(l).Layers() {
            if i == 0 {
                sublayer.SetDeltas(deltaBuffers)
                // undo that last reverse
                deltaBuffers.ReverseBuffers() 
            } else {
                sublayer.SetDeltas(nestedDeltas)
            }
        }
    }

    // Special case if originating from a branch node
    be := backends.Be()
    if toBranchNode(n.prevLayer) != nil {
        prevDeltas := n.prevLayer.Deltas()
        delta := be.Iobuf(n.inShape, nil, base.DtypeNone, "", true, prevDeltas[0])
        n.deltas = []backends.Tensor{delta}
    } else {
        delta := be.Iobuf(n.inShape, nil, base.DtypeNone, "", true, bottomBuffer)
        n.deltas = []backends.Tensor{delta}
        deltaBuffers.ReverseBuffers()
    }
}

func(n *Broadcast) GetTerminal() []Layer {
    t := make([]Layer, len(n.layers))
    for i, l := range n.layers {
        // all layers are Sequential or non-containers; expected to return one terminal
        t[i] = l.GetTerminal()[0]
    }
    return t
}

//
//    Following implementation of MergeSum and MergeBroadcast is suitable for
//    CPU and GPU backends only. Original implemenation of MKL backend requires
//    more complex backend hooks ([fprop|bprop]_mergesum, [fprop|bprop]_mergebroadcast)
//    exposing Layer interface to Backend. Porting this design to Go would require
//    some artificial tricks to avoid circular package dependencies. As we have no
//    intent to support MKL at this phase we eliminate these hooks altogether by
//    inlining common CPU/GPU code for these hooks directly here.
//
//    Also, MergeSum allocate method relies on backend allocate_new_outputs hook
//    for special MKL support. For same reasons as above we eliminate  this hook
//    replacing it with direct call to Layer allocate.
//
//    (These issues shall disappear after transition to proper IR-based design.)
//

//
//    MergeSum
//

type MergeSum struct {
    Broadcast
}

var mergeSumInitArgMap = base.ArgMap{
    "name": base.NewAnyArgOpt(""), // passthru
}

func NewMergeSum(layers []LayerItem, args ...interface{}) *MergeSum {
    n := new(MergeSum)
    n.Init(n, layers, base.MakeArgs(args))
    return n
}

func(n *MergeSum) Init(self base.Object, layers []LayerItem, args base.Args) {
    args = mergeSumInitArgMap.Expand(args)
    n.Broadcast.Init(self, layers, args.Filter([]string{"name"}))
}

func(n *MergeSum) ClassName() string {
    return "arhat.layers.MergsSum"
}

func(n *MergeSum) Configure(inObj InputObject) {
    n.Broadcast.Configure(inObj)
    n.configureMerge()
}

func(n *MergeSum) configureMerge() {
    n.outShape = n.layers[0].OutShape()
}

func(n *MergeSum) Allocate(sharedOutputs backends.Tensor, accumulateUpdates bool) {
    be := backends.Be()
    if n.outputs == nil {
        output := be.Iobuf(n.outShape, nil, base.DtypeNone, "", false, sharedOutputs)
        n.outputs = []backends.Tensor{output}
    }

    for _, l := range n.layers {
        l.Allocate(n.outputs[0], false)
    }
}

func(n *MergeSum) Fprop(
        inputs []backends.Tensor, inference bool, beta float64) []backends.Tensor {
    for i, l := range n.layers {
        b := 1.0
        if i == 0 {
            b = 0.0
        }
        l.Fprop(inputs, inference, b)
    }
    return n.outputs
}

func(n *MergeSum) Bprop(
        errors []backends.Tensor, alpha float64, beta float64) []backends.Tensor {
    last := len(n.layers) - 1
    for i := last; i >= 0; i-- {
        l := n.layers[i]
        b := 1.0
        if i == last {
            b = beta
        }
        l.Bprop(errors, alpha, b)
    }
    return n.deltas
}

//
//    MergeBroadcast
//

type Merge int

const (
    MergeRecurrent Merge = iota
    MergeDepth
    MergeStack
)

var mergeEnum = base.EnumDef{
    "recurrent": int(MergeRecurrent),
    "depth": int(MergeDepth),
    "stack": int(MergeStack),
}

type MergeBroadcast struct {
    Broadcast
    merge Merge
    alphas []float64
    betas []float64
    errorViews []backends.Tensor
    slices [][2]int
}

var mergeBroadcastInitArgMap = base.ArgMap{
    "name": base.NewAnyArgOpt(""), // passthru
    "merge": base.NewEnumArg(mergeEnum),
    "alphas": base.NewFloatListArgOpt(nil),
}

func NewMergeBroadcast(layers []LayerItem, args ...interface{}) *MergeBroadcast {
    n := new(MergeBroadcast)
    n.Init(n, layers, base.MakeArgs(args))
    return n
}

func(n *MergeBroadcast) Init(self base.Object, layers []LayerItem, args base.Args) {
    args = mergeBroadcastInitArgMap.Expand(args)
    n.Broadcast.Init(self, layers, args.Filter([]string{"name"}))
    layerNum := len(layers)
    n.betas = make([]float64,  layerNum)
    for i := 0; i < layerNum - 1; i++ {
        n.betas[i] = 1.0
    }
    n.betas[layerNum-1] = 0.0
    n.alphas = base.ToFloatList(args["alphas"])
    if n.alphas == nil {
        n.alphas = make([]float64, layerNum)
        for i := 0; i < layerNum; i++ {
            n.alphas[i] = 1.0
        }
    }
    n.merge = Merge(args["merge"].(int))
    n.errorViews = nil
}

func(n *MergeBroadcast) ClassName() string {
    return "arhat.layers.MergeBroadcast"
}

func(n *MergeBroadcast) Configure(inObj InputObject) {
    n.Broadcast.Configure(inObj)
    n.configureMerge()
}

func(n *MergeBroadcast) configureMerge() {
    // figure out how to merge
    switch n.merge {
    case MergeRecurrent:
        n.configureMergeRecurrent()
    case MergeDepth:
        n.configureMergeDepth()
    case MergeStack:
        n.configureMergeStack()
    }
}

func(n *MergeBroadcast) configureMergeRecurrent() {
    n.slices = make([][2]int, len(n.layers))
    strideSize := backends.Be().Bsz()
    sumCatdims := 0
    startIdx := 0
    for i, l := range n.layers {
        inShape := l.OutShape()
        catdim := inShape[1]
        sumCatdims += catdim
        endIdx := sumCatdims * strideSize
        n.slices[i][0] = startIdx
        n.slices[i][1] = endIdx
        startIdx = endIdx
    }
    inShape0 := n.layers[0].OutShape()
    n.outShape = []int{inShape0[0], sumCatdims}
}

func(n *MergeBroadcast) configureMergeDepth() {
    n.slices = make([][2]int, len(n.layers))
    inShape0 := n.layers[0].OutShape()
    strideSize := base.IntsProd(inShape0[1:])
    sumCatdims := 0
    startIdx := 0
    for i, l := range n.layers {
        inShape := l.OutShape()
        catdim := inShape[0]
        sumCatdims += catdim
        endIdx := sumCatdims * strideSize
        n.slices[i][0] = startIdx
        n.slices[i][1] = endIdx
        startIdx = endIdx
    }
    n.outShape = make([]int, len(inShape0))
    n.outShape[0] = sumCatdims
    copy(n.outShape[1:], inShape0[1:])
}

func(n *MergeBroadcast) configureMergeStack() {
    n.slices = make([][2]int, len(n.layers))
    strideSize := 1
    sumCatdims := 0
    startIdx := 0
    for i, l := range n.layers {
        inShape := l.OutShape()
        catdim := base.IntsProd(inShape)
        sumCatdims += catdim
        endIdx := sumCatdims * strideSize
        n.slices[i][0] = startIdx
        n.slices[i][1] = endIdx
        startIdx = endIdx
    }
    n.outShape = []int{sumCatdims}
}

func(n *MergeBroadcast) Allocate(sharedOutputs backends.Tensor, accumulateUpdates bool) {
    be := backends.Be()
    if n.outputs == nil {
        output := be.Iobuf(n.outShape, nil, base.DtypeNone, "", false, sharedOutputs)
        n.outputs = []backends.Tensor{output}
    }
    outputViews := n.getPartitions(n.outputs[0], n.slices)
    for i, l := range n.layers {
        l.Allocate(outputViews[i], false)
    }
}

func(n *MergeBroadcast) getPartitions(x backends.Tensor, slices [][2]int) []backends.Tensor {
    result := make([]backends.Tensor, len(slices))
    xshape := x.Shape()
    // is this sequential case?
    sequential := (xshape[len(xshape)-1] != backends.Be().Bsz())
    for i, sl := range slices {
        if sequential {
            result[i] = x.GetItem(backends.MakeSlice(nil, sl[:]))
        } else {
            result[i] = x.GetItem(backends.MakeSlice(sl[:]))
        }
    }
    return result
}

func(n *MergeBroadcast) Fprop(
        inputs []backends.Tensor, inference bool, beta float64) []backends.Tensor {
    for _, l := range n.layers {
        l.Fprop(inputs, inference, 0.0)
    }
    return n.outputs
}

func(n *MergeBroadcast) Bprop(
        errors []backends.Tensor, alpha float64, beta float64) []backends.Tensor {
    if n.errorViews == nil {
        n.errorViews = n.getPartitions(errors[0], n.slices)
    }
    last := len(n.layers) - 1
    for i := last; i >= 0; i-- {
        l := n.layers[i]
        e := n.errorViews[i:i+1]
        a := n.alphas[i]
        b := n.betas[i]
        if i == last {
            b = beta
        }
        l.Bprop(e, a*alpha, b)
    }
    return n.deltas
}

//
//    MergeMultiStream
//

type MergeMultiStream struct {
    MergeBroadcast
}

var mergeMultiStreamInitArgMap = base.ArgMap{
    "name": base.NewAnyArgOpt(""), // passthru
    "merge": base.NewAnyArg(),     // passthru
}

func NewMergeMultiStream(layers []LayerItem, args ...interface{}) *MergeMultiStream {
    n := new(MergeMultiStream)
    n.Init(n, layers, base.MakeArgs(args))
    return n
}

func(n *MergeMultiStream) Init(self base.Object, layers []LayerItem, args base.Args) {
    args = mergeMultiStreamInitArgMap.Expand(args)
    n.MergeBroadcast.Init(self, layers, args.Filter([]string{"name", "merge"}))
}

func(n *MergeMultiStream) NestDeltas() bool {
    return false
}

func(n *MergeMultiStream) ClassName() string {
    return "arhat.layers.MergeMultiStream"
}

/* ACHTUNG: It is not clear how it was supposed to work in original code: inObj is not multi-shape
func(n *MergeMultiStream) Configure(inObj InputObject) {
    n.prevLayer = nil
    // ACHTUNG: If inObj is array of layers, is it correct to replace layers by shapes here?
    multiShape := inObj.MultiShape()
    for i, l := range n.layers {
        l.Configure(MakeInputObject(multiShape[i]))
    }
    n.configureMerge()
}
*/

func(n *MergeMultiStream) Configure(inObj InputObject) {
    // ACHTUNG: Original code requires inObj be list of shapes, one per branch
    //     It is not clear how this was supposed to work: normally we receive
    //     just one regular inObj from parent Sequential layer
    n.prevLayer = nil
    for _, l := range n.layers {
        l.Configure(inObj)
    }
    n.configureMerge()
}

func (n *MergeMultiStream) SetDeltas(deltaBuffers *DeltasTree) {
    // deltaBuffers ignored here, will generate
    // new delta buffers for each sequential container
    for _, l := range n.layers {
        l.AllocateDeltas(nil)
    }
}

func(n *MergeMultiStream) Fprop(
        inputs []backends.Tensor, inference bool, beta float64) []backends.Tensor {
    // ACHTUNG: Original code suggests multiple inputs, one input per layer in n.layers
    //     It is not clear how this could happen: normally we get just one input
    //     and this same input has to be used for all layers
    assertSingle(inputs)
    for _, l := range n.layers {
        l.Fprop(inputs, inference, 0.0)
    }
    return n.outputs
}

func(n *MergeMultiStream) Bprop(
        errors []backends.Tensor, alpha float64, beta float64) []backends.Tensor {
    assertSingle(errors)
    if n.errorViews == nil {
        n.errorViews = n.getPartitions(errors[0], n.slices)
    }
    for i, l := range n.layers {
        l.Bprop(n.errorViews[i:i+1], 1.0, 0.0)
    }
    // output is never used, discard
    return nil
}

//
//    SKIPPED: Encoder
//

//
//    SKIPPED: Decoder
//

//
//    SKIPPED: Seq2Seq
//

//
//    Multicost
//

type Multicost struct {
    base.ObjectBase
    costs []Cost
    weights []float64
    costFunc transforms.Cost
    deltas []backends.Tensor
    inputs []backends.Tensor
}

var multicostInitArgMap = base.ArgMap{
    "name": base.NewAnyArgOpt(""), // passthru
    "weights": base.NewFloatListArgOpt(nil),
}

func NewMulticost(costs []Cost, args ...interface{}) *Multicost {
    n := new(Multicost)
    n.Init(n, costs, base.MakeArgs(args))
    return n
}

func(n *Multicost) Init(self base.Object, costs []Cost, args base.Args) {
    args = multicostInitArgMap.Expand(args)
    n.ObjectBase.Init(self, args.Filter([]string{"name"}))
    n.costs = costs
    n.weights = base.ToFloatList(args["weights"])
    if n.weights == nil {
        costNum := len(costs)
        n.weights = make([]float64, costNum)
        for i := 0; i < costNum; i++ {
            n.weights[i] = 1.0
        }
    }
    n.deltas = nil
    n.inputs = nil
    n.costFunc = costs[0].CostFunc()
}

func(n *Multicost) Initialize(inObj Layer) {
    numCosts := len(n.costs)
    terminals := inObj.GetTerminal()
    base.Assert(len(terminals) == numCosts)
    for i := 0; i < numCosts; i++ {
        c := n.costs[i]
        ll := terminals[i]
        c.Initialize(ll)
    }
}

func(n *Multicost) ClassName() string {
    return "arhat.layers.Multicost"
}

func(n *Multicost) CostFunc() transforms.Cost {
    return n.costFunc
}

func(n *Multicost) Outputs() []backends.Tensor {
    return n.costs[0].Outputs()
}

func(n *Multicost) Deltas() []backends.Tensor {
    return n.deltas
}

func(n *Multicost) GetCost(inputs []backends.Tensor, targets []backends.Tensor) backends.Value {
    be := backends.Be()
    numCosts := len(n.costs)
    base.Assert(len(inputs) == numCosts)
    numTargets := len(targets)
    base.Assert(numTargets == 1 || numTargets == numCosts)
    base.Assert(len(n.weights) == numCosts)
    base.Assert(len(n.costs) == numCosts)
    if numCosts == 1 {
        return n.costs[0].GetCost(inputs, targets)
    }
    var sum backends.Value
    for i := 0; i < numCosts; i++ {
        w := n.weights[i]
        c := n.costs[i]
        x := inputs[i]
        var t backends.Tensor
        if numTargets == 1 {
            t = targets[0]
        } else {
            t = targets[i]
        }
        // TODO(orig): use sentinel class instead of nil
        // it is important that we don't even call get_cost on costs
        // which aren't applicable because there are hooks and state
        // that get set that we don't want to include.
        if t != nil {
            cost := c.GetCost([]backends.Tensor{x}, []backends.Tensor{t})
            cost = be.Float(w).Mul(cost)
            if sum != nil {
                sum = sum.Add(cost)
            } else {
                sum = cost
            }
        }
    }
    return sum
}

func(n *Multicost) GetErrors(
        inputs []backends.Tensor, targets []backends.Tensor) []backends.Tensor {
    be := backends.Be()
    numCosts := len(n.costs)
    base.Assert(len(inputs) == numCosts)
    numTargets := len(targets)
    base.Assert(numTargets == 1 || numTargets == numCosts)
    base.Assert(len(n.weights) == numCosts)
    base.Assert(len(n.costs) == numCosts)
    for i := 0; i < numCosts; i++ {
        var t backends.Tensor
        if numTargets == 1 {
            t = targets[0]
        } else {
            t = targets[i]
        }
        if t == nil {
            continue
        }
        cost := n.costs[i]
        x := inputs[i]
        w := n.weights[i]
        cost.GetErrors([]backends.Tensor{x}, []backends.Tensor{t})
        deltas := n.costs[i].Deltas()
        // don't allow multicosts of multicosts in this release
        assertSingle(deltas)
        deltas[0].Assign(deltas[0].Mul(be.Float(w)))
    }
    if n.deltas == nil {
        n.deltas = make([]backends.Tensor, numCosts)
        for i := 0; i < numCosts; i++ {
            deltas := n.costs[i].Deltas()
            n.deltas[i] = deltas[0]
        }
    }
    return n.deltas
}

//
//    SKIPPED: SkipThought
//

