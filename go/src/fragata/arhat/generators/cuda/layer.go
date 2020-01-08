//
// Copyright 2019-2020 FRAGATA COMPUTER SYSTEMS AG
// Copyright 2014-2018 Intel Corporation
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
    "strings"
)

//
//    Dirty trick to pass batch sum from ConvLayer to subsquent BatchNorm
//
//    TODO: Find better solution not using global variables
//

var lastBsum backends.Tensor

//
//    Layer
//

type Layer interface {
    Dtype() base.Dtype
    N() int
    Flops() float64
    SizeI() int
    SizeF() int
    SizeO() int
    // optional properties
    C() int
    K() int
    M() int
    P() int
    Q() int
    DimI() []int
    DimF() []int
    DimO() []int
    DimI2() []int
    DimF2() []int
    DimO2() []int
    NOut() int
    // setters
    SetBpropOut(bpropOut backends.Tensor)
    SetDeltaStats(deltaStats backends.Tensor)
    // operational methods
    InitActivations(fpropOut backends.Tensor)
    InitDeltas(shared []backends.Tensor)
    InitWeights(loc float64, scale float64, shared backends.Tensor, zeros bool)
    InitDataZero()
    InitDataUniform(low float64, high float64)
    Fprop(fpropIn backends.Tensor, scaleWeights float64) backends.Tensor
    Bprop(bpropIn backends.Tensor, beta float64) backends.Tensor
    String() string
}

//
//    LayerBase
//

type LayerBase struct {
    self Layer
    dtype base.Dtype
    n int
    dtypeU base.Dtype
    lib *CudaGenerator
    flops float64
    sizeI int
    sizeO int
    sizeF int
    weights backends.Tensor
    fpropIn backends.Tensor
    fpropOut backends.Tensor
    bpropIn backends.Tensor
    bpropOut backends.Tensor
    updateOut backends.Tensor
    learningRate float64
    actStats backends.Tensor
    deltaStats backends.Tensor
    weightStats backends.Tensor
}

func(n *LayerBase) Init(
        self Layer, 
        lib *CudaGenerator, 
        dtype base.Dtype, 
        N int, 
        dtypeU base.Dtype) {
    n.self = self
    n.dtype = dtype
    n.n = N
    n.dtypeU = base.ResolveDtype(dtypeU, dtype)
    n.lib = lib
    n.flops = 0
    n.sizeI = 0
    n.sizeO = 0
    n.sizeF = 0
    n.weights = nil
    n.fpropIn = nil
    n.fpropOut = nil
    n.bpropIn = nil
    n.bpropOut = nil
    n.updateOut = nil
    n.learningRate = 0.0
    n.actStats = nil
    n.deltaStats = nil
    n.weightStats = nil
}

func(n *LayerBase) Dtype() base.Dtype { return n.dtype }
func(n *LayerBase) N() int { return n.n }
func(n *LayerBase) Flops() float64 { return n.flops }
func(n *LayerBase) SizeI() int { return n.sizeI }
func(n *LayerBase) SizeF() int { return n.sizeF }
func(n *LayerBase) SizeO() int { return n.sizeO }

func(n *LayerBase) C() int { return n.undefInt() }
func(n *LayerBase) K() int { return n.undefInt() }
func(n *LayerBase) M() int { return n.undefInt() }
func(n *LayerBase) P() int { return n.undefInt() }
func(n *LayerBase) Q() int { return n.undefInt() }
func(n *LayerBase) DimI() []int { return n.undefDim() }
func(n *LayerBase) DimF() []int { return n.undefDim() }
func(n *LayerBase) DimO() []int { return n.undefDim() }
func(n *LayerBase) DimI2() []int { return n.undefDim() }
func(n *LayerBase) DimF2() []int { return n.undefDim() }
func(n *LayerBase) DimO2() []int { return n.undefDim() }
func(n *LayerBase) NOut() int { return n.undefInt() }

func(n *LayerBase) undefInt() int {
    base.Assert(false)
    return 0
}

func(n *LayerBase) undefDim() []int {
    base.Assert(false)
    return nil
}

func(n *LayerBase) SetBpropOut(bpropOut backends.Tensor) { n.bpropOut = bpropOut }
func(n *LayerBase) SetDeltaStats(deltaStats backends.Tensor) { n.deltaStats = deltaStats }

func(n *LayerBase) InitActivations(fpropOut backends.Tensor) {
    if fpropOut != nil {
        n.fpropOut = fpropOut
    } else {
        n.fpropOut = n.lib.NewTensor(n.self.DimO(), n.dtype)
    }
    n.actStats = n.lib.NewTensor([]int{n.self.DimO2()[0], 1}, base.Float32)
}

func(n *LayerBase) InitDeltas(shared []backends.Tensor) {
    if shared == nil {
        n.bpropOut = n.lib.NewTensor(n.self.DimI(), n.dtype)
    } else {
        n.bpropOut = shared[0].Share(n.self.DimI(), base.DtypeNone, "")
        i := 0
        k := len(shared) - 1
        for i < k {
            shared[i], shared[k] = shared[k], shared[i]
            i++
            k--
        }
    }
    n.deltaStats = n.lib.NewTensor([]int{n.self.DimI2()[0], 1}, base.Float32)
}

func(n *LayerBase) InitWeights(loc float64, scale float64, shared backends.Tensor, zeros bool) {
    if n.sizeF > 0 {
        n.weights = n.lib.NewTensor(n.self.DimF(), n.dtype)
        if zeros {
            n.weights.Fill(0.0)
        } else {
            n.lib.RngNormal(n.weights, loc, scale, n.weights.Shape())
        }
        if shared == nil {
            n.updateOut = n.lib.NewTensor(n.self.DimF(), n.dtypeU)
        } else {
            n.updateOut = shared.Share(n.self.DimF(), n.dtypeU, "")
        }
        n.weightStats = n.lib.NewTensor([]int{n.self.DimF2()[0], 1}, base.Float32)
    }
}

/*
    def scale_weights(self, scale):
        mean = self.get_activation_mean()
        self.weights[:] *= scale / mean 

*/

func(n *LayerBase) ScaleWeights(scale float64) {
    // TODO
    base.NotImplementedError()
}

// fprop relu happens inside of the conv and gemm kernels
func(n *LayerBase) BpropRelu(bpropIn backends.Tensor) backends.Tensor {
    // bpropIn[] = bpropIn * (n.fpropOut > 0)
    zero := n.lib.Float(0.0)
    bpropIn.Assign(bpropIn.Mul(n.fpropOut.Gt(zero)))
    return bpropIn
}

func(n *LayerBase) GradDescent() {
    // n.weights[] += n.updateOut * n.learningRate 
    learningRate := n.lib.Float(n.learningRate)
    n.weights.Assign(n.weights.Add(n.updateOut.Mul(learningRate)))
}

func(n *LayerBase) InitDataZero() {
    // defined for DataLayer only
    base.Assert(false)
}

func(n *LayerBase) InitDataUniform(low float64, high float64) {
    // defined for DataLayer only
    base.Assert(false)
}

func(n *LayerBase) Fprop(fpropIn backends.Tensor, scaleWeights float64) backends.Tensor {
    if n.fpropIn == nil && fpropIn != nil {
        n.fpropIn = fpropIn.Reshape(n.self.DimI())
    }
    return n.fpropIn
}

func(n *LayerBase) Bprop(bpropIn backends.Tensor, beta float64) backends.Tensor {
    return bpropIn
}

//
//    DataParams
//

type DataParams struct {
    N int
    C int
    D int
    H int
    W int
}

func(a *DataParams) Init() {
    a.N = base.IntNone
    a.C = base.IntNone
    a.D = base.IntNone
    a.H = base.IntNone
    a.W = base.IntNone
}

func(a *DataParams) Resolve() {
    base.Assert(
        a.N != base.IntNone &&
        a.C != base.IntNone)
    a.D = base.ResolveInt(a.D, 1)
    a.H = base.ResolveInt(a.H, 1)
    a.W = base.ResolveInt(a.W, 1)
}

//
//    DataLayer
//

type DataLayer struct {
    LayerBase
    c int
    k int
    m int
    p int
    q int
    dhw []int
    dimI []int
    dimO []int
    dimI2 []int
    dimO2 []int
}

func NewDataLayer(
        lib *CudaGenerator, 
        dtype base.Dtype, 
        params *DataParams) *DataLayer {
    n := new(DataLayer)
    n.Init(n, lib, dtype, params)
    return n
}

func(n *DataLayer) Init(
        self Layer,
        lib *CudaGenerator, 
        dtype base.Dtype, 
        params *DataParams) {
    a := *params
    a.Resolve()
    n.LayerBase.Init(self, lib, dtype, a.N, base.DtypeNone)
    n.c = a.C
    n.k = a.C
    n.m = a.D
    n.p = a.H
    n.q = a.W
    n.dhw = []int{a.D, a.H, a.W}
    n.dimI = []int{a.C, a.D, a.H, a.W, a.N}
    n.dimO = []int{a.C, a.D, a.H, a.W, a.N}
    n.dimI2 = []int{a.C*a.D*a.H*a.W, a.N}
    n.dimO2 = []int{a.C*a.D*a.H*a.W, a.N}
    n.sizeO = base.IntsProd(n.dimO)
    n.sizeI = n.sizeO
}

func(n *DataLayer) C() int { return n.c }
func(n *DataLayer) K() int { return n.k }
func(n *DataLayer) M() int { return n.m }
func(n *DataLayer) P() int { return n.p }
func(n *DataLayer) Q() int { return n.q }
func(n *DataLayer) DHW() []int { return n.dhw }
func(n *DataLayer) DimI() []int { return n.dimI }
func(n *DataLayer) DimO() []int { return n.dimO }
func(n *DataLayer) DimI2() []int { return n.dimI2 }
func(n *DataLayer) DimO2() []int { return n.dimO2 }

func(n *DataLayer) InitDataZero() {
    n.fpropOut.Fill(0.0)
}

func(n *DataLayer) InitDataUniform(low float64, high float64) {
    n.lib.RngUniform(n.fpropOut, low, high, n.fpropOut.Shape())
}

func(n *DataLayer) InitDeltas(shared []backends.Tensor) {
    // nothing to do
}

func(n *DataLayer) InitWeights(loc float64, scale float64, shared backends.Tensor, zeros bool) {
    // nothing to do
}

func(n *DataLayer) Fprop(fpropIn backends.Tensor, scaleWeights float64) backends.Tensor {
    return n.fpropOut
}

func(n *DataLayer) String() string {
    return fmt.Sprintf("DataLayer: NCK: (%d, %d, %d) DHW: (%d %d %d)", 
        n.n, n.c, n.k, n.dhw[0], n.dhw[1], n.dhw[2])
}

//
//    FullParams
//

type FullParams struct {
    N int
    NIn int
    NOut int
    Relu bool
}

func(a *FullParams) Init() {
    a.N = base.IntNone
    a.NIn = base.IntNone
    a.NOut = base.IntNone
    a.Relu = false
}

func(a *FullParams) Resolve() {
    base.Assert(
        a.N != base.IntNone &&
        a.NIn != base.IntNone &&
        a.NOut != base.IntNone)
}

//
//    FullLayer
//

type FullLayer struct {
    LayerBase
    nIn int
    nOut int
    dimI []int
    dimI2 []int
    dimO []int
    dimO2 []int
    dimF []int
    dimF2 []int
    relu bool
}
 
func NewFullLayer(
        lib *CudaGenerator, 
        dtype base.Dtype, 
        params *FullParams) *FullLayer {
    n := new(FullLayer)
    n.Init(n, lib, dtype, params)
    return n
}

func(n *FullLayer) Init(
        self Layer,
        lib *CudaGenerator, 
        dtype base.Dtype, 
        params *FullParams) {
    a := *params
    a.Resolve()
    n.LayerBase.Init(self, lib, dtype, a.N, base.DtypeNone)
    n.nIn = a.NIn
    n.nOut = a.NOut
    n.flops = float64(a.N) * float64(a.NIn) * float64(a.NOut) * 2.0
    n.dimI = []int{a.NIn, a.N}
    n.dimI2 = []int{a.NIn, a.N}
    n.dimO = []int{a.NOut, a.N}
    n.dimO2 = []int{a.NOut, a.N}
    n.dimF = []int{a.NOut, a.NIn}
    n.dimF2 = []int{a.NOut, a.NIn}
    n.sizeI = a.NIn  * a.N
    n.sizeO = a.NOut * a.N
    n.sizeF = a.NIn  * a.NOut
    n.relu = a.Relu
}

func(n *FullLayer) NIn() int { return n.nIn }
func(n *FullLayer) NOut() int { return n.nOut }
func(n *FullLayer) DimI() []int { return n.dimI }
func(n *FullLayer) DimI2() []int { return n.dimI2 }
func(n *FullLayer) DimO() []int { return n.dimO }
func(n *FullLayer) DimO2() []int { return n.dimO2 }
func(n *FullLayer) DimF() []int { return n.dimF }
func(n *FullLayer) DimF2() []int { return n.dimF2 }
func(n *FullLayer) Relu() bool { return n.relu }

func(n *FullLayer) Fprop(fpropIn backends.Tensor, scaleWeights float64) backends.Tensor {
    fpropIn = n.LayerBase.Fprop(fpropIn, 0.0)
    // ACHTUNG: Apparent bug in original code (misplaced relu argument)
    n.lib.CompoundDot(n.weights, fpropIn, n.fpropOut, 1.0, 0.0, n.relu, nil)

    if scaleWeights != 0.0 {
        n.ScaleWeights(scaleWeights)
        n.Fprop(fpropIn, 0.0)
    }

    return n.fpropOut
}

func(n *FullLayer) Bprop(bpropIn backends.Tensor, beta float64) backends.Tensor {
    if n.relu {
        n.BpropRelu(bpropIn)
    }
    n.lib.CompoundDot(n.weights.T(), bpropIn, n.bpropOut, 1.0, 0.0, false, nil)

    n.lib.CompoundDot(bpropIn, n.fpropIn.T(), n.updateOut, 1.0, 0.0, false, nil)
    n.GradDescent()

    return n.bpropOut
}

func(n *FullLayer) String() string {
    return fmt.Sprintf("FullLayer: N, nIn, nOut: (%d, %d, %d)", n.n, n.nIn, n.nOut)
}

//
//    ConvParams
//

type ConvParams struct {
    N int
    C int 
    K int
    D int
    H int 
    W int
    T int
    R int 
    S int
    M int
    P int
    Q int
    PadD int
    PadH int 
    PadW int
    StrD int 
    StrH int 
    StrW int
    DilD int 
    DilH int 
    DilW int
}

func(a *ConvParams) InitConv(params *backends.ConvParams) {
    p := *params
    p.Resolve()
    a.N = p.N
    a.C = p.C
    a.K = p.K
    a.D = p.D
    a.H = p.H
    a.W = p.W
    a.T = p.T
    a.R = p.R
    a.S = p.S
    a.M = base.IntNone
    a.P = base.IntNone
    a.Q = base.IntNone
    a.PadD = p.PadD
    a.PadH = p.PadH
    a.PadW = p.PadW
    a.StrD = p.StrD
    a.StrH = p.StrH
    a.StrW = p.StrW
    a.DilD = p.DilD
    a.DilH = p.DilH
    a.DilW = p.DilW
}

func(a *ConvParams) InitDeconv(params *backends.DeconvParams) {
    p := *params
    p.Resolve()
    a.N = p.N
    a.C = p.C
    a.K = p.K
    a.D = base.IntNone
    a.H = base.IntNone
    a.W = base.IntNone
    a.T = p.T
    a.R = p.R
    a.S = p.S
    a.M = p.M
    a.P = p.P
    a.Q = p.Q
    a.PadD = p.PadD
    a.PadH = p.PadH
    a.PadW = p.PadW
    a.StrD = p.StrD
    a.StrH = p.StrH
    a.StrW = p.StrW
    a.DilD = p.DilD
    a.DilH = p.DilH
    a.DilW = p.DilW
}

//
//    ConvLayerBase
//

type ConvKernelBuilder interface {
    BuildConvKernels(dtype base.Dtype, a *ConvParams) (
        ConvFpropKernels, ConvBpropKernels, ConvUpdateKernels)
}

type ConvLayerBase struct {
    LayerBase
    c int
    k int
    m int
    p int
    q int
    nck []int
    trs []int
    dhw []int
    mpq []int
    padding []int
    strides []int
    dimI []int
    dimF []int
    dimFb []int
    dimO []int
    dimI2 []int
    dimF2 []int
    dimF2t []int
    dimO2 []int
    dimS []int
    nOut int
    fpropKernels ConvFpropKernels
    bpropKernels ConvBpropKernels
    updateKernels ConvUpdateKernels
    // local extensions (benchmarking only)
    relu bool
    bsum bool
    batchSum backends.Tensor
}

func(n *ConvLayerBase) Init(
        self Layer,
        lib *CudaGenerator, 
        dtype base.Dtype, 
        params *ConvParams,
        kernelBuilder ConvKernelBuilder) {
    a := params

    n.LayerBase.Init(self, lib, dtype, a.N, base.Float32)

    // Compute the output spatial dimensions
    a.M = lib.OutputDim(a.D, a.T, a.PadD, a.StrD, false, a.DilD)
    a.P = lib.OutputDim(a.H, a.R, a.PadH, a.StrH, false, a.DilH)
    a.Q = lib.OutputDim(a.W, a.S, a.PadW, a.StrW, false, a.DilW)

    n.c = a.C
    n.k = a.K
    n.m = a.M
    n.p = a.P
    n.q = a.Q
    n.nck = []int{a.N, a.C, a.K}
    n.trs = []int{a.T, a.R, a.S}
    n.dhw = []int{a.D, a.H, a.W}
    n.mpq = []int{a.M, a.P, a.Q}
    n.padding = []int{a.PadD, a.PadH, a.PadW}
    n.strides = []int{a.StrD, a.StrH, a.StrW}

    n.dimI = []int{a.C, a.D, a.H, a.W, a.N}
    n.dimF = []int{a.C, a.T, a.R, a.S, a.K}
    n.dimFb = []int{a.K, a.T, a.R, a.S, a.C}
    n.dimO = []int{a.K, a.M, a.P, a.Q, a.N}
    n.dimI2 = []int{a.C*a.D*a.H*a.W, a.N}
    n.dimF2 = []int{a.C*a.T*a.R*a.S, a.K}
    n.dimF2t = []int{a.K, a.C*a.T*a.R*a.S}
    n.dimO2 = []int{a.K*a.M*a.P*a.Q, a.N}
    n.dimS = []int{a.K, 1}
    n.sizeI = base.IntsProd(n.dimI)
    n.sizeF = base.IntsProd(n.dimF)
    n.sizeO = base.IntsProd(n.dimO)
    n.nOut = base.IntsProd(n.mpq) * a.K

    // flop count for benchmarking
    n.flops = 
        float64(a.P) * 
        float64(a.Q) * 
        float64(a.M) * 
        float64(a.K) * 
        float64(a.N) * 
        float64(a.C) * 
        float64(a.R) * 
        float64(a.S) * 
        float64(a.T) * 2.0

    // TODO: Apparent bug in origial code; analyze and fix
    dilatedConv := (a.DilD != 1 || a.DilH != 1 || a.DilW != 1)
    if dilatedConv {
        base.Assert(a.DilD > 0 && a.DilH > 0 && a.DilW > 0)
    }

    // Cuda C
    // 3D conv not supported yet
    if a.T > 1 || a.D > 1 {
        base.ValueError("3D Convolution not supported by CUDA C kernels")
    }

    n.fpropKernels, n.bpropKernels, n.updateKernels = kernelBuilder.BuildConvKernels(n.dtype, a)
}

func(n *ConvLayerBase) C() int { return n.c }
func(n *ConvLayerBase) K() int { return n.k }
func(n *ConvLayerBase) M() int { return n.m }
func(n *ConvLayerBase) P() int { return n.p }
func(n *ConvLayerBase) Q() int { return n.q }
func(n *ConvLayerBase) NCK() []int { return n.nck }
func(n *ConvLayerBase) TRS() []int { return n.trs }
func(n *ConvLayerBase) DHW() []int { return n.dhw }
func(n *ConvLayerBase) MPQ() []int { return n.mpq }
func(n *ConvLayerBase) Padding() []int { return n.padding }
func(n *ConvLayerBase) Strides() []int { return n.strides }
func(n *ConvLayerBase) DimI() []int { return n.dimI }
func(n *ConvLayerBase) DimF() []int { return n.dimF }
func(n *ConvLayerBase) DimFb() []int { return n.dimFb } // not in backends.Layer
func(n *ConvLayerBase) DimO() []int { return n.dimO }
func(n *ConvLayerBase) DimI2() []int { return n.dimI2 }
func(n *ConvLayerBase) DimF2() []int { return n.dimF2 }
func(n *ConvLayerBase) DimF2t() []int { return n.dimF2t } // not in backends.Layer
func(n *ConvLayerBase) DimO2() []int { return n.dimO2 }
func(n *ConvLayerBase) DimS() []int { return n.dimS }
func(n *ConvLayerBase) NOut() int { return n.nOut }

func(n *ConvLayerBase) SetRelu(relu bool) { n.relu = relu }
func(n *ConvLayerBase) SetBsum(bsum bool) { n.bsum = bsum }

func(n *ConvLayerBase) InitActivations(fpropOut backends.Tensor) {
    n.LayerBase.InitActivations(fpropOut)

    if n.bsum {
        n.batchSum = n.lib.NewTensor(n.dimS, base.Float32)
    } else {
        n.batchSum = nil
    }
}

func(n *ConvLayerBase) Fprop(fpropIn backends.Tensor, scaleWeights float64) backends.Tensor {
    fpropIn = n.LayerBase.Fprop(fpropIn, 0.0)
    layer := n.self.(backends.ConvLayer)
    n.lib.FpropConv(
        layer,      // layer
        fpropIn,    // i
        n.weights,  // f
        n.fpropOut, // o
        nil,        // x
        nil,        // bias
        n.batchSum, // bsum
        1.0,        // alpha
        0.0,        // beta
        false,      // relu
        false,      // brelu
        0.0)        // slope

    if scaleWeights != 0.0 {
        n.ScaleWeights(scaleWeights)
        n.Fprop(fpropIn, 0.0)
    }

    // ACHTUNG: Original code returns pair (n.fpropOut, n.batchSum) if n.bsum is set
    //     We use global variable instead

    if n.bsum {
        lastBsum = n.batchSum
    } else {
        lastBsum = nil
    }

    return n.fpropOut
}

func(n *ConvLayerBase) Bprop(bpropIn backends.Tensor, beta float64) backends.Tensor {
    if n.relu  {
        n.BpropRelu(bpropIn)
    }
    layer := n.self.(backends.ConvLayer)
    if n.bpropOut != nil {
        n.lib.BpropConv(
            layer,      // layer
            n.weights,  // f
            bpropIn,    // e
            n.bpropOut, // gradI
            nil,        // x
            nil,        // bias
            nil,        // bsum
            1.0,        // alpha
            beta,       // beta
            false,      // relu
            false,      // brelu,
            0.0)        // slope
    }

    n.lib.UpdateConv(
        layer,       // layer
        n.fpropIn,   // i
        bpropIn,     // e
        n.updateOut, // gradF
        1.0,         // alpha
        0.0,         // beta
        nil)         // gradBias
    n.GradDescent()

    return n.bpropOut
}

//
//    ConvLayer
//

type ConvLayer struct {
    ConvLayerBase
}

func NewConvLayer(
        lib *CudaGenerator, 
        dtype base.Dtype, 
        params *backends.ConvParams,
        kernelBuilder ConvKernelBuilder) *ConvLayer {
    n := new(ConvLayer)
    n.Init(n, lib, dtype, params, kernelBuilder)
    return n
}

func(n *ConvLayer) Init(
        self Layer,
        lib *CudaGenerator, 
        dtype base.Dtype, 
        params *backends.ConvParams,
        kernelBuilder ConvKernelBuilder) {
    var a ConvParams
    a.InitConv(params)
    n.ConvLayerBase.Init(self, lib, dtype, &a, kernelBuilder)
}

func(n *ConvLayer) String() string {
    return fmt.Sprintf("ConvLayer: NCK: (%d, %d, %d) HW: (%d, %d)",
        n.n, n.c, n.k, n.dhw[1], n.dhw[2]) 
}

//
//    DeconvLayer
//

type DeconvLayer struct {
    ConvLayerBase
}

func NewDeconvLayer(
        lib *CudaGenerator, 
        dtype base.Dtype, 
        params *backends.DeconvParams,
        kernelBuilder ConvKernelBuilder) *DeconvLayer {
    n := new(DeconvLayer)
    n.Init(n, lib, dtype, params, kernelBuilder)
    return n
}

func(n *DeconvLayer) Init(
        self Layer,
        lib *CudaGenerator, 
        dtype base.Dtype, 
        params *backends.DeconvParams,
        kernelBuilder ConvKernelBuilder) {
    var a ConvParams
    a.InitDeconv(params)

    tt := a.DilD * (a.T - 1) + 1
    rr := a.DilH * (a.R - 1) + 1
    ss := a.DilW * (a.S - 1) + 1

    // Cannot get exact, e.g. because not unique
    a.D = (a.M - 1) * a.StrD - 2 * a.PadD + tt
    a.H = (a.P - 1) * a.StrH - 2 * a.PadH + rr
    a.W = (a.Q - 1) * a.StrW - 2 * a.PadW + ss

    n.ConvLayerBase.Init(self, lib, dtype, &a, kernelBuilder)

    n.nOut = base.IntsProd(n.dhw) * a.C
}

func(n *DeconvLayer) String() string {
    return fmt.Sprintf(
        "DeconvLayer: NCK: (%d, %d, %d) DHW: (%d, %d, %d) TRS: (%d, %d, %d) MPQ: (%d, %d, %d)",
            n.n, n.c, n.k, 
            n.dhw[0], n.dhw[1], n.dhw[2], 
            n.trs[0], n.trs[1], n.trs[2], 
            n.mpq[0], n.mpq[1], n.mpq[2])
}

//
//    PoolLayer
//

type PoolKernel struct {
    name string
    grid []int
    block []int
    args []int
}

func NewPoolKernel(
        name string, 
        grid []int,
        block []int, 
        args ...interface{}) *PoolKernel {
    return &PoolKernel{name, grid, block, Flatten(args...)}
}

type PoolLayer struct {
    LayerBase
    op backends.PoolOp
    c int
    k int
    m int
    p int
    q int
    jtrs []int
    dhw []int
    mpq []int
    padding []int
    strides []int
    dimI []int
    dimO []int
    dimF2 []int
    dimI2 []int
    dimO2 []int
    nOut int
    overlap bool // ACHTUNG: Originally float, no idea why
/* ACHTUNG: Effectively unused
    gaps int
*/
    fpropKernel *PoolKernel
    bpropKernel *PoolKernel
    fpropLutSize int
    bpropLutSize int
    // local extensions (benchmarking only)
    argmax backends.Tensor
}

func NewPoolLayer(
        lib *CudaGenerator, 
        dtype base.Dtype, 
        params *backends.PoolParams) *PoolLayer {
    n := new(PoolLayer)
    n.Init(n, lib, dtype, params)
    return n
}

var sbLarge = map[int][]int{
//  SB        shlP maskP shrP shlQ maskQ shrQ maskN shrN
    1:  []int{0,   0x00, 0,   0,   0x00, 0,   0xfff, 32}, // 1x1  nnnnn
    2:  []int{0,   0x00, 0,   1,   0x10, 4,   0x00f,  4}, // 1x2  xnnnn
    4:  []int{0,   0x00, 0,   2,   0x18, 3,   0x007,  3}, // 1x4  xxnnn
    8:  []int{0,   0x00, 0,   3,   0x1c, 2,   0x003,  2}, // 1x8  xxxnn
    16: []int{0,   0x00, 0,   4,   0x1e, 1,   0x001,  1}, // 1x16 xxxxn
    32: []int{0,   0x00, 0,   5,   0x1f, 0,   0x000,  0}, // 1x32 xxxxx
}

var sbMedium = map[int][]int{
//  SB        shlP maskP shrP shlQ maskQ shrQ maskN shrN
    8:  []int{1,   0x10, 4,   2,   0x0c, 2,   0x003,  2}, // 2x4  yxxnn
    16: []int{1,   0x10, 4,   3,   0x0e, 1,   0x001,  1}, // 2x8  yxxxn
    32: []int{1,   0x10, 4,   4,   0x0f, 0,   0x000,  0}, // 2x16 yxxxx
}

var sbSmall = map[int][]int{
//  SB        shlP maskP shrP shlQ maskQ shrQ maskN shrN
    16: []int{2,   0x18, 3,   2,   0x06, 1,   0x001,  1}, // 4x4  yyxxn
    32: []int{2,   0x18, 3,   3,   0x07, 0,   0x000,  0}, // 4x8  yyxxx
}

func(n *PoolLayer) Init(
        self Layer,
        lib *CudaGenerator, 
        dtype base.Dtype, 
        params *backends.PoolParams) {
    a := *params
    a.Resolve()

    n.LayerBase.Init(self, lib, dtype, a.N, base.DtypeNone)

/* ACHTUNG: Unused (retained from withdrawn support for non-float32 data?)
    var class string
    switch n.dtype {
    case base.Float16:
        class = "hpool"
    case base.Float32:
        class = "spool"
    default:
        base.TypeError("Type not supported.")
    }
*/

    // default to non-overlapping
    a.StrC = base.ResolveInt(a.StrC, a.J)
    a.StrD = base.ResolveInt(a.StrD, a.T)
    a.StrH = base.ResolveInt(a.StrH, a.R)
    a.StrW = base.ResolveInt(a.StrW, a.S)

    // SKIPPED (orig): Computation of n.overlap

/* TODO: Revise this
    n.overlap = 1.0
*/
    n.overlap = true

/* ACHTUNG: Effectively unused
    // TODO(orig): detect other forms of gaps
    if a.StrC > a.J || a.StrD > a.T || a.StrH > a.R || a.StrW > a.S {
        n.gaps = 1
    } else {
        n.gaps = 0
    }
*/

/* ACHTUNG: Unused
    bpropZero := (n.overlap != 0.0 || n.gaps != 0)
*/

    // compute the output dimensions
    k := lib.OutputDim(a.C, a.J, a.PadC, a.StrC, true, 1)
    m := lib.OutputDim(a.D, a.T, a.PadD, a.StrD, true, 1)
    p := lib.OutputDim(a.H, a.R, a.PadH, a.StrH, true, 1)
    q := lib.OutputDim(a.W, a.S, a.PadW, a.StrW, true, 1)

    n.op = a.Op
    n.c = a.C
    n.k = k
    n.m = m
    n.p = p
    n.q = q
    n.jtrs = []int{a.J, a.T, a.R, a.S}
    n.dhw = []int{a.D, a.H, a.W}
    n.mpq = []int{m, p, q}
    n.padding = []int{a.PadC, a.PadD, a.PadH, a.PadW}
    n.strides = []int{a.StrC, a.StrD, a.StrH, a.StrW}

    n.dimI = []int{a.C, a.D, a.H, a.W, a.N}
    n.dimO = []int{k, m, p, q, a.N}
    n.dimF2 = nil
    n.dimI2 = []int{a.C*a.D*a.H*a.W, a.N}
    n.dimO2 = []int{k*m*p*q, a.N}
    n.sizeI = base.IntsProd(n.dimI)
    n.sizeO = base.IntsProd(n.dimO)
    n.nOut = base.IntsProd(n.mpq) * k

    // precompute some multiplications for fast constant memory access
    wn := a.W * a.N
    hwn := a.H * wn
    dhwn := a.D * hwn
    dh := a.D * a.H
    rs := a.R * a.S
    rst := a.T * rs
    jrst := a.J * rst
    qn := q * a.N
    pqn := p * qn
    mpqn := m * pqn

    base.AssertMsg(jrst + 32 < (1 << 16), "Integer division is faster with 16bit numerators")

    var superBlock int
    switch {
    case a.N == 1:
        superBlock = 0
    case a.N < 32:
        superBlock = bitCount(a.N-1)
    default:
        superBlock = 5
    }
    superBlock = 1 << uint(5-superBlock)

    // try to minimize the zero overlap in the superblock
    // but maximize the x dim of the superblock for more contiguous memory access
    var sbParams []int
    switch {
    case superBlock < 8 || q > 64:
        sbParams = sbLarge[superBlock]
    case superBlock < 16 || q > 32:
        sbParams = sbMedium[superBlock]
    default:
        sbParams = sbSmall[superBlock]
    }

    supP := ceilDiv(p, 1<<uint(sbParams[0]))
    supQ := ceilDiv(q, 1<<uint(sbParams[3]))

    // precompute the magic numbers and shift amounts for integer division
    magicRst := magic32(jrst+32, rst)
    magicRs := magic32(rst+32, rs)
    magicS := magic32(rs+32, a.S)
    magicP := magic32(m*supP, supP)

    fpropName := "fprop_" + a.Op.String()
    bpropName := "bprop_" + a.Op.String()

    threads := a.N
    if superBlock > 1 {
        threads = 32
    }

    n.fpropKernel = 
        NewPoolKernel(
            fpropName, 
            []int{supQ, supP*m, k}, 
            []int{threads, 1, 1},
            a.N, 
            a.W, 
            a.H, 
            a.D, 
            a.C, 
            wn, 
            hwn, 
            dhwn,
            p, 
            q, 
            magicP, 
            qn, 
            pqn, 
            mpqn,
            a.PadC, 
            a.PadD, 
            a.PadH, 
            a.PadW,
            a.StrC, 
            a.StrD, 
            a.StrH, 
            a.StrW,
            a.S, 
            rs, 
            rst, 
            jrst, 
            magicS, 
            magicRs, 
            magicRst,
            supP, 
            supQ, 
            sbParams)

    lutSize := jrst
    if lutSize % 4 != 0 {
        lutSize += 4 - lutSize % 4
    }

    n.fpropLutSize = superBlock * lutSize * 4
    n.bpropLutSize = n.fpropLutSize

/* TODO: Revise this
    if n.overlap > 0.0 {
*/
    if n.overlap {

        // we have a special kernel to handle the overlapping avg pooling
        bpropName += "_overlap"

        magicStrW := magic32(a.W+a.S, a.StrW)
        magicStrH := magic32(a.H+a.R, a.StrH)
        magicStrD := magic32(a.D+a.T, a.StrD)
        magicStrC := magic32(a.C+a.J, a.StrC)

        if superBlock > 1 {

            bpropName += "_smallN"

            // TODO: Encapsulate this repetitive code in a closure
            switch {
            case superBlock < 8 || a.W > 64:
                sbParams = sbLarge[superBlock]
            case superBlock < 16 || a.W > 32:
                sbParams = sbMedium[superBlock]
            default:
                sbParams = sbSmall[superBlock]
            }

            supH := ceilDiv(a.H, 1<<uint(sbParams[0]))
            supW := ceilDiv(a.W, 1<<uint(sbParams[3]))

            magicH := magic32(a.D*supH, supH)

            maxLutSize := 
                ceilDiv(a.S, a.StrW) *
                ceilDiv(a.R, a.StrH) *
                ceilDiv(a.T, a.StrD) *
                ceilDiv(a.J, a.StrC)

            n.bpropKernel = 
                NewPoolKernel(
                    bpropName, 
                    []int{supW, a.D*supH, a.C}, 
                    []int{threads, 1, 1},
                    a.N, 
                    a.W, 
                    a.H, 
                    a.D, 
                    a.C, 
                    wn, 
                    hwn, 
                    dhwn, 
                    magicH,
                    a.PadW, 
                    a.PadH, 
                    a.PadD, 
                    a.PadC,
                    a.StrW, 
                    a.StrH, 
                    a.StrD, 
                    a.StrC,
                    magicStrW, 
                    magicStrH, 
                    magicStrD, 
                    magicStrC,
                    a.S, 
                    a.R, 
                    a.T, 
                    a.J, 
                    rs, 
                    rst, 
                    jrst, 
                    magicS, 
                    magicRs, 
                    magicRst,
                    q, 
                    p, 
                    m, 
                    k, 
                    qn, 
                    pqn, 
                    mpqn,
                    supH, 
                    supW, 
                    sbParams, 
                    maxLutSize)

            lutSize = maxLutSize
            if lutSize % 4 != 0 {
                lutSize += 4 - lutSize % 4
            }

            n.bpropLutSize = superBlock * lutSize * 4 * 2

        } else {

            // The overlap kernel can be much more efficient if we aren't doing superblocking
            magicH := magic32(dh, a.H)

            n.bpropKernel = 
                NewPoolKernel(
                    bpropName, 
                    []int{a.W, dh, a.C}, 
                    []int{threads, 1, 1},
                    a.N, 
                    a.W, 
                    a.H, 
                    a.D, 
                    a.C, 
                    wn, 
                    hwn, 
                    dhwn, 
                    magicH,
                    a.PadW, 
                    a.PadH, 
                    a.PadD, 
                    a.PadC,
                    a.StrW, 
                    a.StrH, 
                    a.StrD, 
                    a.StrC,
                    magicStrW, 
                    magicStrH, 
                    magicStrD, 
                    magicStrC,
                    a.S, 
                    a.R, 
                    a.T, 
                    a.J, 
                    rs, 
                    rst, 
                    jrst,
                    magicS, 
                    magicRs, 
                    magicRst,
                    q, 
                    p, 
                    m, 
                    k, 
                    qn, 
                    pqn, 
                    mpqn)

                n.bpropLutSize = lutSize * 4 * 2
        }

    } else {

        // ACHTUNG: This branch is never used as n.overlap is fixed to true
        //     (same in original code)

        n.bpropKernel = 
            NewPoolKernel(
                bpropName, 
                []int{supQ, supP*m, k}, 
                []int{threads, 1, 1},
                a.N, 
                a.W, 
                a.H, 
                a.D, 
                a.C, 
                wn, 
                hwn, 
                dhwn,
                p, 
                q, 
                magicP, 
                qn, 
                pqn, 
                mpqn,
                a.PadC, 
                a.PadD, 
                a.PadH, 
                a.PadW,
                a.StrC, 
                a.StrD, 
                a.StrH, 
                a.StrW,
                a.S, 
                rs, 
                rst, 
                jrst, 
                magicS, 
                magicRs, 
                magicRst,
                supP, 
                supQ, 
                sbParams)
    }
}

func(n *PoolLayer) Op() backends.PoolOp { return n.op }
func(n *PoolLayer) C() int { return n.c }
func(n *PoolLayer) K() int { return n.k }
func(n *PoolLayer) M() int { return n.m }
func(n *PoolLayer) P() int { return n.p }
func(n *PoolLayer) Q() int { return n.q }
func(n *PoolLayer) JTRS() []int { return n.jtrs }
func(n *PoolLayer) DHW() []int { return n.dhw }
func(n *PoolLayer) MPQ() []int { return n.mpq }
func(n *PoolLayer) Padding() []int { return n.padding }
func(n *PoolLayer) Strides() []int { return n.strides }
func(n *PoolLayer) DimI() []int { return n.dimI }
func(n *PoolLayer) DimO() []int { return n.dimO }
func(n *PoolLayer) DimF2() []int { return n.dimF2 }
func(n *PoolLayer) DimI2() []int { return n.dimI2 }
func(n *PoolLayer) DimO2() []int { return n.dimO2 }
func(n *PoolLayer) NOut() int { return n.nOut }

func(n *PoolLayer) InitActivations(fpropOut backends.Tensor) {
    n.LayerBase.InitActivations(fpropOut)
    n.argmax = n.lib.NewTensor(n.dimO, base.Uint8)
}

func(n *PoolLayer) Fprop(fpropIn backends.Tensor, scaleWeights float64) backends.Tensor {
    fpropIn = n.LayerBase.Fprop(fpropIn, 0.0)
    n.lib.FpropPool(
        n,          // layer
        fpropIn,    // i
        n.fpropOut, // o
        n.argmax,   // argmax
        1.0,        // alpha
        0.0)        // beta
    return n.fpropOut
}

func(n *PoolLayer) Bprop(bpropIn backends.Tensor, beta float64) backends.Tensor {
    // ACHTUNG: Argument beta is not used in original code: why?
    n.lib.BpropPool(
        n,          // layer
        bpropIn,    // i
        n.bpropOut, // o
        n.argmax,   // argmax
        1.0,        // alpha
        0.0)        // beta
    return n.bpropOut
}

func(n *PoolLayer) String() string {
    return fmt.Sprintf(
        "PoolLayer: NCK: (%d, %d, %d) DHW: (%d, %d, %d) "+
        "JTRS: (%d, %d, %d, %d) MPQ: (%d, %d, %d) op: %s",
            n.n, n.c, n.k,
            n.dhw[0], n.dhw[1], n.dhw[2],
            n.jtrs[0], n.jtrs[1], n.jtrs[2], n.jtrs[3],
            n.mpq[0], n.mpq[1], n.mpq[2],
            n.op)
}

//
//    LrnLayer
//

type LrnLayer struct {
    PoolLayer
}

func NewLrnLayer(
        lib *CudaGenerator,
        dtype base.Dtype, 
        params *backends.LrnParams) *LrnLayer {
    n := new(LrnLayer)
    n.Init(n, lib, dtype, params)
    return n
}

func(n *LrnLayer) Init(
        self Layer,
        lib *CudaGenerator,
        dtype base.Dtype, 
        params *backends.LrnParams) {
    a := *params
    a.Resolve()
    base.AssertMsg(a.J % 2 == 1, "Only support odd LRN window size")
    padC := a.J / 2
    var p backends.PoolParams
    p.Init(backends.PoolOpLrn)
    p.N = a.N
    p.C = a.C
    p.D = a.D
    p.H = a.H
    p.W = a.W
    p.J = a.J
    p.T = 1
    p.R = 1
    p.S = 1
    p.PadC = padC
    p.PadD = 0
    p.PadH = 0
    p.PadW = 0
    p.StrC = 1
    p.StrD = 1
    p.StrH = 1
    p.StrW = 1
    n.PoolLayer.Init(self, lib, dtype, &p)
}

//
//    InceptionParams
//

type InceptionParams struct {
    N int
    C int
    K int
    D int
    H int
    W int
    M int
    P int
    Q int
}

func(a *InceptionParams) Init() {
    a.N = base.IntNone
    a.C = base.IntNone
    a.K = base.IntNone
    a.D = base.IntNone
    a.H = base.IntNone
    a.W = base.IntNone
    a.M = base.IntNone
    a.P = base.IntNone
    a.Q = base.IntNone
}

func(a *InceptionParams) Resolve() {
    base.Assert(
        a.N != base.IntNone &&
        a.C != base.IntNone &&
        a.K != base.IntNone)
    a.D = base.ResolveInt(a.D, 1)
    a.H = base.ResolveInt(a.H, 1)
    a.W = base.ResolveInt(a.W, 1)
    a.M = base.ResolveInt(a.M, 1)
    a.P = base.ResolveInt(a.P, 1)
    a.Q = base.ResolveInt(a.Q, 1)
}

//
//    InceptionLayer
//

type InceptionLayer struct {
    LayerBase
    partitions [][]Layer
    c int
    k int
    m int
    p int
    q int
    nck []int
    dhw []int
    mpq []int
    dimI []int
    dimO []int
    dimI2 []int
    dimO2 []int
    dimF []int
    nOut int
}

func NewInceptionLayer(
        lib *CudaGenerator, 
        dtype base.Dtype, 
        partitions [][]Layer,
        params *InceptionParams) *InceptionLayer {
    n := new(InceptionLayer)
    n.Init(n, lib, dtype, partitions, params)
    return n
}

func(n *InceptionLayer) Init(
        self Layer,
        lib *CudaGenerator, 
        dtype base.Dtype, 
        partitions [][]Layer,
        params *InceptionParams) {
    a := *params
    a.Resolve()

    n.LayerBase.Init(self, lib, dtype, a.N, base.DtypeNone)

    n.partitions = partitions

    n.c = a.C
    n.k = a.K
    n.m = a.M
    n.p = a.P
    n.q = a.Q
    n.nck = []int{a.N, a.C, a.K}
    n.dhw = []int{a.D, a.H, a.W}
    n.mpq = []int{a.M, a.P, a.Q}

    n.dimI  = []int{a.C, a.D, a.H, a.W, a.N}
    n.dimO  = []int{a.K, a.M, a.P, a.Q, a.N}
    n.dimI2 = []int{a.C * a.D * a.H * a.W, a.N}
    n.dimO2 = []int{a.K * a.M * a.P * a.Q, a.N}
    n.sizeI = base.IntsProd(n.dimI)
    n.sizeO = base.IntsProd(n.dimO)
    n.nOut  = base.IntsProd(n.mpq) * a.K

    n.sizeF = 0
    n.flops = 0

    for _, part := range partitions {
        for _, layer := range part {
            n.flops += layer.Flops()
            n.sizeF = base.IntMax(n.sizeF, layer.SizeF())
            // skip layers with zero filter size to avoid DimF panic
            if n.sizeF != 0 && n.sizeF == layer.SizeF() {
                n.dimF = layer.DimF()
            }
        }
    }
}

func(n *InceptionLayer) Partitions() [][]Layer { return n.partitions }
func(n *InceptionLayer) C() int { return n.c }
func(n *InceptionLayer) K() int { return n.k }
func(n *InceptionLayer) M() int { return n.m }
func(n *InceptionLayer) P() int { return n.p }
func(n *InceptionLayer) Q() int { return n.q }
func(n *InceptionLayer) NCK() []int { return n.nck }
func(n *InceptionLayer) DHW() []int { return n.dhw }
func(n *InceptionLayer) MPQ() []int { return n.mpq }
func(n *InceptionLayer) DimI() []int { return n.dimI }
func(n *InceptionLayer) DimO() []int { return n.dimO }
func(n *InceptionLayer) DimI2() []int { return n.dimI2 }
func(n *InceptionLayer) DimO2() []int { return n.dimO2 }
func(n *InceptionLayer) DimF() []int { return n.dimF }
func(n *InceptionLayer) NOut() int { return n.nOut }

func(n *InceptionLayer) InitActivations(fpropOut backends.Tensor) {
    n.LayerBase.InitActivations(fpropOut)
    k := 0
    for _, part := range n.partitions {
        last := len(part) - 1
        for i, layer := range part {
            if i == last {
                nextK := k+layer.K()
                slice := backends.MakeSlice([]int{k, nextK}, backends.Ellipsis{})
                layer.InitActivations(n.fpropOut.GetItem(slice))
                k = nextK
            } else {
                layer.InitActivations(nil)
            }
        }
    }
}

func(n *InceptionLayer) InitDeltas(shared []backends.Tensor) {
    n.LayerBase.InitDeltas(shared)
    var sharedDeltas []backends.Tensor
    if shared != nil {
        sharedDeltas = shared[1:3]
    }
    for _, part := range n.partitions {
        for i, layer := range part {
            if i == 0 {
                layer.SetBpropOut(n.bpropOut)
                layer.SetDeltaStats(n.deltaStats)
            } else {
                layer.InitDeltas(sharedDeltas)
            }
        }
    }
}

func(n *InceptionLayer) InitWeights(
        loc float64, scale float64, shared backends.Tensor, zeros bool) {
    for _, part := range n.partitions {
        for _, layer := range part {
            layer.InitWeights(loc, scale, shared, zeros)
        }
    }
}

func(n *InceptionLayer) Fprop(fpropIn backends.Tensor, scaleWeights float64) backends.Tensor {
    fpropIn = n.LayerBase.Fprop(fpropIn, 0.0)
    for _, part := range n.partitions {
        partFpropIn := fpropIn
        for _, layer := range part {
            partFpropIn = layer.Fprop(partFpropIn, scaleWeights)
        }
    }
    return n.fpropOut
}

func(n *InceptionLayer) Bprop(bpropIn backends.Tensor, beta float64) backends.Tensor {
    k := n.k
    lastI := len(n.partitions) - 1
    for i := lastI; i >= 0; i-- {
        part :=  n.partitions[i]
        lastP := len(part) - 1
        prevK := k - part[lastP].K()
        slice := backends.MakeSlice([]int{prevK, k}, backends.Ellipsis{})
        partBpropIn := bpropIn.GetItem(slice)
        k = prevK
        for p := lastP; p >= 0; p-- {
            layer := part[p]
            beta := 0.0
            if i != lastI && p == 0 {
                beta = 1.0
            }
            partBpropIn = layer.Bprop(partBpropIn, beta)
        }
    }
    return n.bpropOut
}

// SKIPPED: FpropStats, BpropStats (for the time being)

func(n *InceptionLayer) String() string {
    out := 
        fmt.Sprintf("Inception: NCK: (%d, %d, %d) DHW: (%d, %d, %d) MPQ: (%d, %d, %d)\n",
            n.n, n.c, n.k, n.dhw[0], n.dhw[1], n.dhw[2], n.mpq[0], n.mpq[1], n.mpq[2])
    for i, part := range n.partitions {
        out += fmt.Sprintf("  Part%d:\n", i+1)
        for _, layer := range part {
            out += fmt.Sprintf("    %s\n", layer)
        }
    }
    return strings.TrimRight(out, "\n")
}

/*
    def fprop_stats(self):
        for part in self.partitions:
            for layer in part:
                layer.fprop_stats()

    def bprop_stats(self):
        for part in self.partitions[::-1]:
            for layer in part[::-1]:
                layer.bprop_stats()
*/

//
//    BatchNormParams
//

type BatchNormParams struct {
    N int
    C int
    D int
    H int
    W int
    NIn int
    Rho float64
    Eps float64
    Relu bool
    Bsum bool
}

func(a *BatchNormParams) Init() {
    a.N = base.IntNone
    a.C = base.IntNone
    a.D = base.IntNone
    a.H = base.IntNone
    a.W = base.IntNone
    a.NIn = base.IntNone
    a.Rho = 0.99   // FloatNone and resolve?
    a.Eps = 1.0e-6 // FloatNone and resolve?
    a.Relu = false
    a.Bsum = false
}

func(a *BatchNormParams) Resolve() {
    // C and NIn may be None
    base.Assert(a.N != base.IntNone)
    a.D = base.ResolveInt(a.D, 1)
    a.H = base.ResolveInt(a.H, 1)
    a.W = base.ResolveInt(a.W, 1)
}

//
//    BatchNormLayer
//

type BatchNormLayer struct {
    LayerBase
    rho float64
    eps float64
    relu bool
    bsum bool
    c int
    k int
    m int
    p int
    q int
    dimI []int
    dimO []int
    dimO2 []int
    dim2 []int
    nOut int
    rcpDepth float64
    // local extensions (benchmarking only)
    xvar backends.Tensor
    xsum backends.Tensor
    beta backends.Tensor
    gamma backends.Tensor
    gmean backends.Tensor
    gvar backends.Tensor
    gradBeta backends.Tensor
    gradGamma backends.Tensor
}

func NewBatchNormLayer(
        lib *CudaGenerator, 
        dtype base.Dtype, 
        params *BatchNormParams) *BatchNormLayer {
    n := new(BatchNormLayer)
    n.Init(n, lib, dtype, params)
    return n
}

func(n *BatchNormLayer) Init(
        self Layer,
        lib *CudaGenerator, 
        dtype base.Dtype, 
        params *BatchNormParams) {
    a := *params
    a.Resolve()

    n.LayerBase.Init(self, lib, dtype, a.N, base.DtypeNone)

    n.rho  = a.Rho
    n.eps  = a.Eps
    n.relu = a.Relu
    n.bsum = a.Bsum

    switch {
    case a.C != base.IntNone:
        n.c = a.C
        n.k = a.C
        n.m = a.D
        n.p = a.H
        n.q = a.W
        n.dimI = []int{a.C, a.D, a.H, a.W, a.N}
        n.dimO = []int{a.C, a.D, a.H, a.W, a.N}
        n.dimO2 = []int{a.C * a.D * a.H * a.W, a.N}
        n.dim2 = []int{a.C, a.D * a.H * a.W * a.N}
        n.nOut = a.C * a.D * a.H * a.W

    case a.NIn != base.IntNone:
        n.nOut = a.NIn
        n.k = a.NIn
        n.dimI = []int{a.NIn, a.N}
        n.dimO = []int{a.NIn, a.N}
        n.dimO2 = []int{a.NIn, a.N}
        n.dim2 = []int{a.NIn, a.N}

    default:
        base.ValueError("missing C or nIn")
    }

    n.rcpDepth = 1.0 / float64(n.dim2[1])
}

func(n *BatchNormLayer) Rho() float64 { return n.rho }
func(n *BatchNormLayer) Eps() float64 { return n.eps }
func(n *BatchNormLayer) Relu() bool { return n.relu }
func(n *BatchNormLayer) Bsum() bool { return n.bsum }
func(n *BatchNormLayer) C() int { return n.c }
func(n *BatchNormLayer) K() int { return n.k }
func(n *BatchNormLayer) M() int { return n.m }
func(n *BatchNormLayer) P() int { return n.p }
func(n *BatchNormLayer) Q() int { return n.q }
func(n *BatchNormLayer) DimI() []int { return n.dimI }
func(n *BatchNormLayer) DimO() []int { return n.dimO }
func(n *BatchNormLayer) DimO2() []int { return n.dimO2 }
func(n *BatchNormLayer) Dim2() []int { return n.dim2 }
func(n *BatchNormLayer) NOut() int  { return n.nOut }
func(n *BatchNormLayer) RcpDepth() float64 { return n.rcpDepth }

func(n *BatchNormLayer) InitActivations(fpropOut backends.Tensor) {
    if fpropOut != nil {
        n.fpropOut = fpropOut.Reshape(n.dim2)
    } else {
        n.fpropOut = n.lib.NewTensor(n.dim2, n.dtype)
    }

    n.xvar = n.lib.NewTensor([]int{n.k, 1}, n.dtype)
    if !n.bsum {
        n.xsum = n.lib.NewTensor([]int{n.k, 1}, base.Float32)
    }
}

func(n *BatchNormLayer) InitDeltas(shared []backends.Tensor) {
    // nothing to do
}

func(n *BatchNormLayer) InitWeights(
        loc float64, scale float64, shared backends.Tensor, zeros bool) {
    n.beta = n.lib.NewTensor([]int{n.k, 1}, n.dtype)
    n.gamma = n.lib.NewTensor([]int{n.k, 1}, n.dtype)
    n.beta.Fill(0.0)
    n.gamma.Fill(1.0)

    n.gmean = n.lib.NewTensor([]int{n.k, 1}, n.dtype)
    n.gvar = n.lib.NewTensor([]int{n.k, 1}, n.dtype)
    n.gmean.Fill(0.0)
    n.gvar.Fill(0.0)

    n.gradBeta = n.lib.NewTensor([]int{n.k, 1}, n.dtype)
    n.gradGamma = n.lib.NewTensor([]int{n.k, 1}, n.dtype)
    n.gradBeta.Fill(0.0)
    n.gradGamma.Fill(0.0)
}

func(n *BatchNormLayer) Fprop(fpropIn backends.Tensor, scaleWeights float64) backends.Tensor {
    if n.fpropIn == nil {
        n.fpropIn = fpropIn.Reshape(n.dim2)
    }

    // ACHTUNG: Original code smuggles bsum via fpropIn
    //     We use global variable instead
    //     Also we prefer to use n.bsum in if condition

    if !n.bsum {
        n.xsum.Assign(n.lib.Sum(n.fpropIn, 1))
    } else {
        base.Assert(lastBsum != nil)
        n.xsum = lastBsum
    }

    // ACHTUNG: Argument relu is misplaces in original code
    n.lib.CompoundFpropBn(
        n.fpropIn,  // x
        n.xsum,     // xsum
        n.xvar,     // xvar
        n.gmean,    // gmean
        n.gvar,     // gvar
        n.gamma,    // gamma
        n.beta,     // beta
        n.fpropOut, // y
        n.eps,      // eps
        n.rho,      // rho
        false,      // computeBatchSum,
        0.0,        // accumbeta
        n.relu,     // relu
        false,      // binary
        false,      // inference
        nil,        // outputs
        n)          // layer

    return n.fpropOut
}

/* ACHTUNG: There is apparent bug in original code; below is presumably correct version
func(n *BatchNormLayer) Bprop(bpropIn backends.Tensor, beta float64) backends.Tensor {
    if n.bpropIn == nil {
        n.bpropIn = bpropIn.Reshape(n.dim2)
    }

    if n.relu {
        n.BpropRelu(n.bpropIn)
    }

    n.lib.CompoundBpropBn(
        n.bpropIn,   // deltaOut
        n.gradGamma, // gradGamma
        n.gradBeta,  // gradBeta
        n.fpropIn,   // deltaIn
        // MISSING: x
        n.xsum,      // xsum
        n.xvar,      // xvar
        n.gamma,     // gamma
        n.eps,       // eps
        false,       // binary,
        n)           // layer

    return n.bpropIn
}
*/

func(n *BatchNormLayer) Bprop(bpropIn backends.Tensor, beta float64) backends.Tensor {
    // compute in place, reuse bpropIn (instead of bpropOut)

    if n.bpropIn == nil {
        n.bpropIn = bpropIn.Reshape(n.dim2)
    }

    if n.relu {
        n.BpropRelu(n.bpropIn)
    }

    n.lib.CompoundBpropBn(
        n.bpropIn,   // deltaOut
        n.gradGamma, // gradGamma
        n.gradBeta,  // gradBeta
        n.bpropIn,   // deltaIn
        n.fpropIn,   // x
        n.xsum,      // xsum
        n.xvar,      // xvar
        n.gamma,     // gamma
        n.eps,       // eps
        false,       // binary,
        n)           // layer

    return n.bpropIn
}

func(n *BatchNormLayer) String() string {
    return fmt.Sprintf("BatchNorm: (%d, %d)", n.dim2[0], n.dim2[1])
}

