//
// Copyright 2019 FRAGATA COMPUTER SYSTEMS AG
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

package cpu

import (
    "fragata/arhat/backends"
    "fragata/arhat/base"
    "fragata/arhat/generators/acc"
)

//
//    ConvParams
//

//
//    ACHTUNG: This code is the same as in generators/cuda
//        Apparently it should be the same for all accelerators
//        Move it to generators/acc?
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

type ConvLayerBase struct {
    lib *CpuGenerator
    dtype base.Dtype
    c int
    d int
    h int
    w int
    n int
    t int
    r int
    s int
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
    dilation []int
    dimI []int
    dimF []int
    dimO []int
    dimS []int
    dimI2 []int
    dimF2 []int
    dimO2 []int
    sizeI int
    sizeF int
    sizeO int
    nOut int
    dot bool
}

func(n *ConvLayerBase) Init(lib *CpuGenerator, dtype base.Dtype, params *ConvParams) {
    n.lib = lib
    n.dtype = dtype

    a := params

    // Compute the output spatial dimensions
    m := lib.OutputDim(a.D, a.T, a.PadD, a.StrD, false, a.DilD)
    p := lib.OutputDim(a.H, a.R, a.PadH, a.StrH, false, a.DilH)
    q := lib.OutputDim(a.W, a.S, a.PadW, a.StrW, false, a.DilW)

    n.c = a.C
    n.d = a.D
    n.h = a.H
    n.w = a.W
    n.n = a.N
    n.t = a.T
    n.r = a.R
    n.s = a.S
    n.k = a.K
    n.m = m
    n.p = p
    n.q = q

    n.nck = []int{a.N, a.C, a.K}
    n.trs = []int{a.T, a.R, a.S}
    n.dhw = []int{a.D, a.H, a.W}
    n.mpq = []int{m, p, q}
    n.padding = []int{a.PadD, a.PadH, a.PadW}
    n.strides = []int{a.StrD, a.StrH, a.StrW}
    n.dilation = []int{a.DilD, a.DilH, a.DilW}

    n.dimI = []int{a.C, a.D, a.H, a.W, a.N}
    n.dimF = []int{a.C, a.T, a.R, a.S, a.K}
    n.dimO = []int{a.K, m, p, q, a.N}
    n.dimS = []int{a.K, 1}
    n.dimI2 = []int{a.C * a.D * a.H * a.W, a.N}
    n.dimF2 = []int{a.C * a.T * a.R * a.S, a.K}
    n.dimO2 = []int{a.K * m * p * q, a.N}
    n.sizeI = base.IntsProd(n.dimI)
    n.sizeF = base.IntsProd(n.dimF)
    n.sizeO = base.IntsProd(n.dimO)
    n.nOut = base.IntsProd(n.mpq) * a.K

    if n.trs[0] == 1 && n.trs[1] == 1 && n.trs[2] == 1 &&
            n.padding[0] == 0 && n.padding[1] == 0 && n.padding[2] == 0 &&
            n.strides[0] == 1 && n.strides[1] == 1 && n.strides[2] == 1 &&
            n.dilation[0] == 1 && n.dilation[1] == 1 && n.dilation[2] == 1 {
        n.dot = true
    } else {
        n.dot = false
    }
}

func(n *ConvLayerBase) N() int { return n.n }
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
func(n *ConvLayerBase) DimO() []int { return n.dimO }
func(n *ConvLayerBase) DimI2() []int { return n.dimI2 }
func(n *ConvLayerBase) DimF2() []int { return n.dimF2 }
func(n *ConvLayerBase) DimO2() []int { return n.dimO2 }
func(n *ConvLayerBase) DimS() []int { return n.dimS }
func(n *ConvLayerBase) SizeI() int { return n.sizeI }
func(n *ConvLayerBase) SizeF() int { return n.sizeF }
func(n *ConvLayerBase) SizeO() int { return n.sizeO }
func(n *ConvLayerBase) NOut() int { return n.nOut }

func(n *ConvLayerBase) FpropConv(
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
    if x == nil {
        x = o
    }

    if n.dot {
        // 1x1 conv can be cast as a simple dot operation
        c := n.c
        k := n.k

        // KxHWN = CxK.T . CxHWN
        f2 := f.Reshape([]int{c, k}).T()
        i2 := i.Reshape([]int{c, -1})

        be := n.lib
        // 'o2' shares data buffer with 'o'
        o2 := o.Reshape([]int{k, -1})
        if beta != 0.0 {
            // o[] = alpha * dot(f, i).reshape(o.shape) + beta * x
            x2 := o.Reshape([]int{k, -1})
            o2.Assign(be.Float(alpha).Mul(be.Dot(f2, i2)).Add(be.Float(beta).Mul(x2)))
        } else {
            // beta and compound ops are mutually exclusive
            o2.Assign(be.Float(alpha).Mul(be.Dot(f2, i2)))
            n.CompoundOps(o, x, bias, bsum, relu, brelu, slope)
        }

        return
    }

    if n.dtype != base.Float32 {
        base.TypeError("Unsupported type for cpu convolution")
    }

    falpha := formatFloat32(alpha)
    fbeta := formatFloat32(beta)

    fmtData := func(a acc.DeviceAllocation) string {
        return formatBufferRef(a.(*CpuDeviceAllocation), true)
    }

    idata := fmtData(i.(*acc.AccTensor).AccData())
    fdata := fmtData(f.(*acc.AccTensor).AccData())
    odata := fmtData(o.(*acc.AccTensor).AccData())
    xdata := fmtData(x.(*acc.AccTensor).AccData())

    n.lib.writeLongStmt(
        "FpropConv(%s, %s, (float *)%s, (float *)%s, (float *)%s, (float *)%s,#"+
            "%d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d,#"+
            "%d, %d, %d, %d, %d, %d, %d, %d, %d);",
                falpha, fbeta, idata, fdata, odata, xdata,
                n.c, n.d, n.h, n.w, n.n, n.t, n.r, n.s, n.k, n.m, n.p, n.q,
                n.strides[0], n.strides[1], n.strides[2], 
                n.padding[0], n.padding[1], n.padding[2], 
                n.dilation[0], n.dilation[1], n.dilation[2])

    if beta == 0.0 {
        n.CompoundOps(o, x, bias, bsum, relu, brelu, slope)
    }
}

func(n *ConvLayerBase) BpropConv(
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
    if x == nil {
        x = o
    }

    if n.dot {
        // 1x1 conv can be cast as a simple dot operation
        c := n.c
        k := n.k

        // CxHWN = CxK . KxHWN
        f2 := f.Reshape([]int{c, k})
        i2 := i.Reshape([]int{k, -1})

        be := n.lib
        // 'o2' shares data buffer with 'o'
        o2 := o.Reshape([]int{c, -1})
        if beta != 0.0 {
            // o[] = alpha * dot(F, I).reshape(o.shape) + beta * x
            x2 := o.Reshape([]int{c, -1})
            o2.Assign(be.Float(alpha).Mul(be.Dot(f2, i2)).Add(be.Float(beta).Mul(x2)))
        } else {
            // beta and compound ops are mutually exclusive
            o2.Assign(be.Float(alpha).Mul(be.Dot(f2, i2)))
            n.CompoundOps(o, x, bias, bsum, relu, brelu, slope)
        }

        return
    }

    if n.dtype != base.Float32 {
        base.TypeError("Unsupported type for cpu convolution")
    }

    falpha := formatFloat32(alpha)
    fbeta := formatFloat32(beta)

    fmtData := func(a acc.DeviceAllocation) string {
        return formatBufferRef(a.(*CpuDeviceAllocation), true)
    }

    idata := fmtData(i.(*acc.AccTensor).AccData())
    fdata := fmtData(f.(*acc.AccTensor).AccData())
    odata := fmtData(o.(*acc.AccTensor).AccData())
    xdata := fmtData(x.(*acc.AccTensor).AccData())

    n.lib.writeLongStmt(
        "BpropConv(%s, %s, (float *)%s, (float *)%s, (float *)%s, (float *)%s,#"+
            "%d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d,#"+
            "%d, %d, %d, %d, %d, %d, %d, %d, %d);",
                falpha, fbeta, idata, fdata, odata, xdata,
                n.c, n.d, n.h, n.w, n.n, n.t, n.r, n.s, n.k, n.m, n.p, n.q,
                n.strides[0], n.strides[1], n.strides[2], 
                n.padding[0], n.padding[1], n.padding[2], 
                n.dilation[0], n.dilation[1], n.dilation[2])

    if beta == 0.0 {
        n.CompoundOps(o, x, bias, bsum, relu, brelu, slope)
    }
}

func(n *ConvLayerBase) UpdateConv(
        i backends.Tensor,
        e backends.Tensor,
        u backends.Tensor,
        alpha float64,
        beta float64,
        gradBias backends.Tensor) {
    c := n.c
    k := n.k

    if gradBias != nil {
        be := n.lib
        e2 := e.Reshape([]int{k, -1})
        gradBiasFlat := gradBias.Reshape([]int{-1})
        gradBiasFlat.Assign(be.Sum(e2, 1))
    }

    // 1x1 conv can be cast as a simple dot operation
    if n.dot {
        // CxK = CxHWN . KxHWN.T
        i2 := i.Reshape([]int{c, -1})
        e2 := e.Reshape([]int{k, -1}).T()

        be := n.lib
        u2 := u.Reshape([]int{c, -1})
        if beta != 0.0 {
            // u[] = alpha * dot(i, e).reshape(u.shape) + beta * u
            u.Assign(be.Float(alpha).Mul(be.Dot(i2, e2)).Add(be.Float(beta).Mul(u2)))
        } else {
            u.Assign(be.Float(alpha).Mul(be.Dot(i2, e2)))
        }

        return
    }

    if n.dtype != base.Float32 {
        base.TypeError("Unsupported type for cpu convolution")
    }

    falpha := formatFloat32(alpha)
    fbeta := formatFloat32(beta)

    fmtData := func(a acc.DeviceAllocation) string {
        return formatBufferRef(a.(*CpuDeviceAllocation), true)
    }

    idata := fmtData(i.(*acc.AccTensor).AccData())
    edata := fmtData(e.(*acc.AccTensor).AccData())
    udata := fmtData(u.(*acc.AccTensor).AccData())

    n.lib.writeLongStmt(
        "UpdateConv(%s, %s, (float *)%s, (float *)%s, (float *)%s,#"+
            "%d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d,#"+
            "%d, %d, %d, %d, %d, %d, %d, %d, %d);",
                falpha, fbeta, idata, edata, udata,
                n.c, n.d, n.h, n.w, n.n, n.t, n.r, n.s, n.k, n.m, n.p, n.q,
                n.strides[0], n.strides[1], n.strides[2], 
                n.padding[0], n.padding[1], n.padding[2], 
                n.dilation[0], n.dilation[1], n.dilation[2])
}

func(n *ConvLayerBase) CompoundOps(
        o backends.Tensor,
        x backends.Tensor,
        bias backends.Tensor,
        bsum backends.Tensor,
        relu bool,
        brelu bool,
        slope float64) {
    be := n.lib
    oShape := o.Shape()
    if bias != nil {
        // o[] = (o.reshape((o.shape[0], -1)) + bias).reshape(o.shape)
        o2 := o.Reshape([]int{oShape[0], -1})
        o2.Assign(o2.Add(bias))
    }
    switch {
    case relu:
        // o[] = maximum(o, 0) + slope * minimum(o, 0)
        zero := be.Float(0.0)
        if slope != 0.0 {
            o.Assign(be.Maximum(o, zero).Add(be.Float(slope).Mul(be.Minimum(o, zero))))
        } else {
            o.Assign(be.Maximum(o, zero))
        }
    case brelu:
        // o[] = o * ((x > 0) + slope * (x < 0))
        zero := be.Float(0.0)
        if slope != 0.0 {
            o.Assign(o.Mul(x.Gt(zero).Add(be.Float(slope).Mul(x.Lt(zero)))))
        } else {
            o.Assign(o.Mul(x.Gt(zero)))
        }
    }
    if bsum != nil {
        // bsum[] = sum(o.reshape((o.shape[0], -1)), axis=1)
        o2 := o.Reshape([]int{oShape[0], -1})
        bsumFlat := bsum.Reshape([]int{-1})
        bsumFlat.Assign(be.Sum(o2, 1))
    }
}

//
//    ConvLayer
//

type ConvLayer struct {
    ConvLayerBase
}

func NewConvLayer(
        lib *CpuGenerator, 
        dtype base.Dtype, 
        params *backends.ConvParams) *ConvLayer {
    n := new(ConvLayer)
    n.Init(lib, dtype, params)
    return n
}

func(n *ConvLayer) Init(
        lib *CpuGenerator, 
        dtype base.Dtype, 
        params *backends.ConvParams) {
    var a ConvParams
    a.InitConv(params)
    n.ConvLayerBase.Init(lib, dtype, &a)
}

//
//    DeconvLayer
//

type DeconvLayer struct {
    ConvLayerBase
}

func NewDeconvLayer(
        lib *CpuGenerator, 
        dtype base.Dtype, 
        params *backends.DeconvParams) *DeconvLayer {
    n := new(DeconvLayer)
    n.Init(lib, dtype, params)
    return n
}

func(n *DeconvLayer) Init(
        lib *CpuGenerator, 
        dtype base.Dtype, 
        params *backends.DeconvParams) {
    var a ConvParams
    a.InitDeconv(params)

    tt := a.DilD * (a.T - 1) + 1
    rr := a.DilH * (a.R - 1) + 1
    ss := a.DilW * (a.S - 1) + 1

    // Cannot get exact, e.g. because not unique
    a.D = (a.M - 1) * a.StrD - 2 * a.PadD + tt
    a.H = (a.P - 1) * a.StrH - 2 * a.PadH + rr
    a.W = (a.Q - 1) * a.StrW - 2 * a.PadW + ss

    n.ConvLayerBase.Init(lib, dtype, &a)

    n.nOut = base.IntsProd(n.dhw) * a.C
}

//
//    PoolLayer
//

type PoolLayer struct {
    lib *CpuGenerator
    dtype base.Dtype
    op backends.PoolOp
    n int
    c int
    d int
    h int
    w int
    j int
    t int
    r int
    s int
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
    sizeI int
    sizeO int
    nOut int
}

func NewPoolLayer(lib *CpuGenerator, dtype base.Dtype, params *backends.PoolParams) *PoolLayer {
    n := new(PoolLayer)
    n.Init(lib, dtype, params)
    return n
}

func(n *PoolLayer) Init(lib *CpuGenerator, dtype base.Dtype, params *backends.PoolParams) {
    n.lib = lib
    n.dtype = dtype

    a := *params
    a.Resolve()

    // default to non-overlapping
    a.StrC = base.ResolveInt(a.StrC, a.J)
    a.StrD = base.ResolveInt(a.StrD, a.T)
    a.StrH = base.ResolveInt(a.StrH, a.R)
    a.StrW = base.ResolveInt(a.StrW, a.S)

    // SKIPPED: n.overlap (apparently never used)

    // compute the output dimensions
    k := lib.OutputDim(a.C, a.J, a.PadC, a.StrC, true, 1)
    m := lib.OutputDim(a.D, a.T, a.PadD, a.StrD, true, 1)
    p := lib.OutputDim(a.H, a.R, a.PadH, a.StrH, true, 1)
    q := lib.OutputDim(a.W, a.S, a.PadW, a.StrW, true, 1)

    n.op = a.Op
    n.n = a.N
    n.c = a.C
    n.d = a.D
    n.h = a.H
    n.w = a.W
    n.j = a.J
    n.t = a.T
    n.r = a.R
    n.s = a.S
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
}

func(n *PoolLayer) Op() backends.PoolOp { return n.op }
func(n *PoolLayer) N() int { return n.n }
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
func(n *PoolLayer) SizeI() int { return n.sizeI }
func(n *PoolLayer) SizeF() int { return 0 } // backends.Layer
func(n *PoolLayer) SizeO() int { return n.sizeO }
func(n *PoolLayer) NOut() int { return n.nOut }

func(n *PoolLayer) FpropPool(
        i backends.Tensor,
        o backends.Tensor,
        argmax backends.Tensor,
        beta float64) {
    if n.dtype != base.Float32 {
        base.TypeError("Unsupported type for cpu pooling")
    }

    var eop string
    switch n.op {
    case backends.PoolOpMax:
        eop = "PoolOp::Max"
    case backends.PoolOpAvg:
        eop = "PoolOp::Avg"
    case backends.PoolOpL2:
        eop = "PoolOp::L2"
    }

    fbeta := formatFloat32(beta)

    fmtData := func(a acc.DeviceAllocation) string {
        return formatBufferRef(a.(*CpuDeviceAllocation), true)
    }

    idata := fmtData(i.(*acc.AccTensor).AccData())
    odata := fmtData(o.(*acc.AccTensor).AccData())
    adata := fmtData(argmax.(*acc.AccTensor).AccData())

    n.lib.writeLongStmt(
        "FpropPool((float *)%s, (float *)%s, (uint8_t *)%s, %s, %s,#"+
            "%d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d,#"+
            "%d, %d, %d, %d, %d, %d, %d, %d);",
                idata, odata, adata, fbeta, eop,
                n.n, n.c, n.d, n.h, n.w, n.j, n.t, n.r, n.s, n.k, n.m, n.p, n.q,
                n.padding[0], n.padding[1], n.padding[2], n.padding[3],
                n.strides[0], n.strides[1], n.strides[2], n.strides[3])
}

func(n *PoolLayer) BpropPool(
        i backends.Tensor,
        o backends.Tensor,
        argmax backends.Tensor,
        alpha float64,
        beta float64) {
    if n.dtype != base.Float32 {
        base.TypeError("Unsupported type for cpu pooling")
    }

    var eop string
    switch n.op {
    case backends.PoolOpMax:
        eop = "PoolOp::Max"
    case backends.PoolOpAvg:
        eop = "PoolOp::Avg"
    case backends.PoolOpL2:
        eop = "PoolOp::L2"
    }

    falpha := formatFloat32(alpha)
    fbeta := formatFloat32(beta)

    fmtData := func(a acc.DeviceAllocation) string {
        return formatBufferRef(a.(*CpuDeviceAllocation), true)
    }

    idata := fmtData(i.(*acc.AccTensor).AccData())
    odata := fmtData(o.(*acc.AccTensor).AccData())
    adata := fmtData(argmax.(*acc.AccTensor).AccData())

    n.lib.writeLongStmt(
        "BpropPool((float *)%s, (float *)%s, (uint8_t *)%s, %s, %s, %s,#"+
            "%d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d,#"+
            "%d, %d, %d, %d, %d, %d, %d, %d);",
                idata, odata, adata, falpha, fbeta, eop,
                n.n, n.c, n.d, n.h, n.w, n.j, n.t, n.r, n.s, n.k, n.m, n.p, n.q,
                n.padding[0], n.padding[1], n.padding[2], n.padding[3],
                n.strides[0], n.strides[1], n.strides[2], n.strides[3])
}

//
//    LrnLayer
//

type LrnLayer struct {
    PoolLayer
}

func NewLrnLayer(
        lib *CpuGenerator,
        dtype base.Dtype, 
        params *backends.LrnParams) *LrnLayer {
    n := new(LrnLayer)
    n.Init(lib, dtype, params)
    return n
}

func(n *LrnLayer) Init(
        lib *CpuGenerator,
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
    n.PoolLayer.Init(lib, dtype, &p)
    base.Assert(
        n.t == 1 &&
        n.r == 1 &&
        n.s == 1 &&
        n.k == n.c && 
        n.m == n.d && 
        n.p == n.h && 
        n.q == n.w &&
        n.padding[0] * 2 + 1 == n.j &&
        n.padding[1] == 0 &&
        n.padding[2] == 0 &&
        n.padding[3] == 0 &&
        n.strides[0] == 1 &&
        n.strides[1] == 1 &&
        n.strides[2] == 1 &&
        n.strides[3] == 1)
}

func(n *LrnLayer) FpropLrn(
        i backends.Tensor,
        o backends.Tensor,
        denom backends.Tensor,
        ascale float64,
        bpower float64) {
    if n.dtype != base.Float32 {
        base.TypeError("Unsupported type for cpu lrn")
    }

    fascale := formatFloat32(ascale)
    fbpower := formatFloat32(bpower)

    fmtData := func(a acc.DeviceAllocation) string {
        return formatBufferRef(a.(*CpuDeviceAllocation), true)
    }

    idata := fmtData(i.(*acc.AccTensor).AccData())
    odata := fmtData(o.(*acc.AccTensor).AccData())
    adata := fmtData(denom.(*acc.AccTensor).AccData())

    n.lib.writeLongStmt(
        "FpropLrn((float *)%s, (float *)%s, (float *)%s,#"+
            "%s, %s, %d, %d, %d, %d, %d, %d);",
                idata, odata, adata,
                fascale, fbpower, n.n, n.c, n.d, n.h, n.w, n.j)
}

func(n *LrnLayer) BpropLrn(
        i backends.Tensor,
        o backends.Tensor,
        e backends.Tensor,
        delta backends.Tensor,
        denom backends.Tensor,
        ascale float64,
        bpower float64) {
    if n.dtype != base.Float32 {
        base.TypeError("Unsupported type for cpu lrn")
    }

    fascale := formatFloat32(ascale)
    fbpower := formatFloat32(bpower)

    fmtData := func(a acc.DeviceAllocation) string {
        return formatBufferRef(a.(*CpuDeviceAllocation), true)
    }

    idata := fmtData(i.(*acc.AccTensor).AccData())
    odata := fmtData(o.(*acc.AccTensor).AccData())
    edata := fmtData(e.(*acc.AccTensor).AccData())
    ddata := fmtData(delta.(*acc.AccTensor).AccData())
    adata := fmtData(denom.(*acc.AccTensor).AccData())

    n.lib.writeLongStmt(
        "BpropLrn((float *)%s, (float *)%s, (float *)%s, (float *)%s, (float *)%s,#"+
            "%s, %s, %d, %d, %d, %d, %d, %d);",
                idata, odata, edata, ddata, adata,
                fascale, fbpower, n.n, n.c, n.d, n.h, n.w, n.j)
}

