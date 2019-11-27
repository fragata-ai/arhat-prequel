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

package cuda

import (
    "fmt"
    "fragata/arhat/backends"
    "fragata/arhat/base"
    "fragata/arhat/generators/acc"
    kernels "fragata/arhat/kernels/cuda"
    "strings"
)

//
//    ConvFpropKernels
//

type ConvFpropKernels interface {
    BindParams(
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
        slope float64)
    Execute()
}

//
//    ConvBpropKernels
//

type ConvBpropKernels interface {
    BindParams(
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
        slope float64)
    Execute()
}

//
//    ConvUpdateKernels
//

type ConvUpdateKernels interface {
    BindParams(
        i backends.Tensor,
        e backends.Tensor,
        o backends.Tensor,
        alpha float64,
        beta float64,
        noOp bool)
    Execute()
}

//
//    KernelGroup
//

type KernelGroup struct {
    params ConvParams
    lib *CudaGenerator
    dtype base.Dtype
    kernelName string
}

func(g *KernelGroup) Init(
        lib *CudaGenerator,
        dtype base.Dtype,
        params *ConvParams) {
    g.params = *params
    g.lib = lib
    g.dtype = dtype
    g.kernelName = ""
}

func(g *KernelGroup) GetKernel(
        dtype base.Dtype,
        filterSize int,
        bsum bool,
        operation string) *Kernel {
    key := fmt.Sprintf("conv_%s_%d_%t_%s_false_false", dtype, filterSize, bsum, operation)
    kernel := g.lib.LookupKernel(key)
    if kernel != nil {
        return kernel
    }
    kernel = g.getKernelRaw(dtype, filterSize, bsum, operation)
    g.lib.RegisterKernel(key, kernel)
    return kernel
}

func(g *KernelGroup) getKernelRaw(
        dtype base.Dtype,
        filterSize int,
        bsum bool,
        operation string) *Kernel {
    name, code := kernels.GetConvKernel(dtype, filterSize, bsum, operation, false, false)
    return NewKernel(g.lib, name, code)
}

//
//    FpropCuda
//

type FpropCuda struct {
    KernelGroup
    rs int
    grid []int
    block []int
    launchArgs []acc.KernelArgument
    staticArgs []int
    shared int
    outputTrans *CompoundOps
}

func NewFpropCuda(
        lib *CudaGenerator, 
        dtype base.Dtype, 
        params *ConvParams) *FpropCuda {
    g := new(FpropCuda)
    g.Init(lib, dtype, params)
    return g
}

func(g *FpropCuda) Init(
        lib *CudaGenerator, 
        dtype base.Dtype, 
        params *ConvParams) {
    g.KernelGroup.Init(lib, dtype, params)

    g.kernelName = "FpropCuda"

    a := params

    base.AssertMsg(a.N % 32 == 0, "N dim must be multiple of 32")

    // ACHTUNG: Is this relevant for CUDA kernels?
    var vecSize int
    if dtype.ItemSize() == 4 {
        vecSize = 4
    } else {
        vecSize = 8
    }
    base.AssertMsg(a.K % vecSize == 0, "K dim must be multiple of %d", vecSize)

    magicPq := magic64(a.P*a.Q)
    magicQ := magic64(a.Q)
    magicS := magic32(a.R*a.S+32, a.S)
    hwn := a.H * a.W * a.N
    rst := a.R * a.S * a.T
    krst := a.K * rst
    pq := a.P * a.Q
    pqn := pq * a.N
    g.rs = a.R * a.S

    g.grid = []int{pq * ceilDiv(a.N, 32), ceilDiv(a.K, 32), 1}
    g.block = []int{8, 8, 1}
    g.staticArgs = 
        Flatten(
            a.C, 
            a.D, 
            a.H, 
            a.W, 
            a.N, 
            a.T, 
            a.R, 
            a.S, 
            a.K, 
            a.M, 
            a.P, 
            a.Q,
            a.StrW, 
            a.StrH, 
            a.PadW, 
            a.PadH,
            a.DilW, 
            a.DilH,
            hwn/4, 
            krst/4, 
            pqn/4,
            pq, 
            0, 
            0,
            magicPq, 
            magicQ, 
            magicS)
    g.shared = rst * 4 * 2

    g.outputTrans = NewCompoundOps(lib, dtype, a.K, pqn)
    lib.SetScratchSize(g.outputTrans.size)
}

func(g *FpropCuda) BindParams(
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
    base.Assert(i.Dtype() == g.dtype && f.Dtype() == g.dtype && o.Dtype() == g.dtype)
    g.lib.ScratchBufferInit()
    betaOrSlope := beta
    if betaOrSlope == 0.0 {
        betaOrSlope = slope
    }
    outputData := g.outputTrans.BindParams(o, x, bias, bsum, alpha, betaOrSlope, relu, brelu)
    // SKIPPED: Stream as launchArgs[0]
    g.launchArgs = []acc.KernelArgument{1.0, 0.0, i, f, outputData, 0}
}

// SKIPPED: Arguments 'repeat' and 'unbind'
func(g *FpropCuda) Execute() {
    kernel := g.GetKernel(g.dtype, g.rs, false, "fprop")
    kernel.Launch(g.grid, g.block, g.shared, g.launchArgs, g.staticArgs)
    g.outputTrans.Execute()
/* SKIPPED
    if unbind {
        g.outputTrans.Unbind()
        g.launchArgs = nil
    }
*/
}

//
//    BpropCuda
//

type BpropCuda struct {
    KernelGroup
    rs int
    grid []int
    block []int
    launchArgs []acc.KernelArgument
    staticArgs []int
    shared int
    filterTrans *FilterDimShuffle
    outputTrans *CompoundOps
}

func NewBpropCuda(
        lib *CudaGenerator, 
        dtype base.Dtype, 
        params *ConvParams) *BpropCuda {
    g := new(BpropCuda)
    g.Init(lib, dtype, params)
    return g
}

func(g *BpropCuda) Init(
        lib *CudaGenerator, 
        dtype base.Dtype, 
        params *ConvParams) {
    g.KernelGroup.Init(lib, dtype, params)
    
    g.kernelName = "BpropCuda"

    a := params

    base.AssertMsg(a.N % 32 == 0, "N dim must be multiple of 32")

    magicHw := magic64(a.H*a.W)
    magicW := magic64(a.W)
/* TODO: Revise this (unused)
    magicRs := magic32(a.R*a.S*a.T+32, a.R*a.S)
*/
    magicS := magic32(a.R*a.S+32, a.S)
    hw := a.H * a.W
    hwn := hw * a.N
    rst := a.R * a.S * a.T
    crst := a.C * rst
    pq := a.P * a.Q
    pqn := pq * a.N
    g.rs = a.R * a.S

    g.grid = []int{hw * ceilDiv(a.N, 32), ceilDiv(a.C, 32), 1}
    g.block = []int{8, 8, 1}
    g.staticArgs = 
        Flatten(
            a.K, 
            a.M, 
            a.P, 
            a.Q, 
            a.N, 
            a.T, 
            a.R, 
            a.S, 
            a.C, 
            a.D, 
            a.H, 
            a.W,
            a.StrW, 
            a.StrH, 
            a.PadW, 
            a.PadH,
            a.DilW, 
            a.DilH,
            pqn/4, 
            crst/4, 
            hwn/4,
            hw, 
            0, 
            0,
            magicHw, 
            magicW, 
            magicS)
    g.shared = a.R * a.S * a.T * 4 * 2

    g.filterTrans = NewFilterDimShuffle(lib, dtype, a.C, a.T, a.R, a.S, a.K)
    g.outputTrans = NewCompoundOps(lib, dtype, a.C, hwn)
    lib.SetScratchSize(g.filterTrans.size, g.outputTrans.size)
}

func(g *BpropCuda) BindParams(
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
    base.Assert(i.Dtype() == g.dtype && f.Dtype() == g.dtype && o.Dtype() == g.dtype)
    base.AssertMsg(g.params.C % 4 == 0, "C dim must be a multiple of 4 for CUDA bprop kernel")

    g.lib.ScratchBufferInit()
    filterData := g.filterTrans.BindParams(f)
    betaOrSlope := beta
    if betaOrSlope == 0.0 {
        betaOrSlope = slope
    }
    outputData := g.outputTrans.BindParams(o, x, bias, bsum, alpha, betaOrSlope, relu, brelu)
    // SKIPPED: Stream as launchArgs[0]
    g.launchArgs = []acc.KernelArgument{1.0, 0.0, i, filterData, outputData, 0}
}

// SKIPPED: Arguments 'repeat' and 'unbind'
func(g *BpropCuda) Execute() {
    kernel := g.GetKernel(g.dtype, g.rs, false, "bprop")
    g.filterTrans.Execute()
    kernel.Launch(g.grid, g.block, g.shared, g.launchArgs, g.staticArgs)
    g.outputTrans.Execute()
/* SKIPPED
    if unbind {
        g.outputTrans.Unbind()
        g.filterTrans.Unbind()
        g.launchArgs = nil
    }
*/
}

//
//    UpdateCuda
//

type UpdateCuda struct {
    KernelGroup
    rs int
    determ int
    grid []int
    block []int
    launchArgs []acc.KernelArgument
    staticArgs []int
    outputTrans *UpdateConvReduce
    outputData acc.DeviceAllocation
    outputDataSize int
}

func NewUpdateCuda(
        lib *CudaGenerator, 
        dtype base.Dtype, 
        params *ConvParams) *UpdateCuda {
    g := new(UpdateCuda)
    g.Init(lib, dtype, params)
    return g
}

func(g *UpdateCuda) Init(
        lib *CudaGenerator, 
        dtype base.Dtype, 
        params *ConvParams) {
    g.KernelGroup.Init(lib, dtype, params)

    g.kernelName = "UpdateData"

    a := params

    base.AssertMsg(a.N % 32 == 0, "N dim must be multiple of 32")

    hwn := a.H * a.W * a.N
    rs := a.R * a.S
    rst := rs * a.T
    krst := a.K * rst
    crstk := krst * a.C
    pq := a.P * a.Q
    pqn := pq * a.N
    magicS := magic32(a.R*a.S+32, a.S)
    g.rs = a.R * a.S

    var gridP, gridQ int
    if lib.Deterministic() {
        gridP = 1
        gridQ = 1
        g.determ = crstk
    } else {
        gridP = a.P
        gridQ = a.Q
        g.determ = 0
    }

    pqBlocks := gridP * gridQ
    magicPq := magic64(pqBlocks)
    magicQ := magic64(gridQ)

    g.grid = []int{pqBlocks * ceilDiv(a.K, 32), ceilDiv(a.C*rs, 32), 1}
    g.block = []int{8, 32, 1}
    g.staticArgs = 
        Flatten(
            a.C, 
            a.D, 
            a.H, 
            a.W, 
            a.N, 
            a.T, 
            a.R, 
            a.S, 
            a.K, 
            a.M, 
            a.P, 
            a.Q,
            a.StrW, 
            a.StrH, 
            a.PadW, 
            a.PadH,
            a.DilW, 
            a.DilH,
            hwn/4, 
            krst/4, 
            pqn/4,
            pqBlocks, 
            gridP, 
            gridQ,
            magicPq, 
            magicQ, 
            magicS)

    g.outputTrans = NewUpdateConvReduce(lib, 1, crstk)
    lib.SetScratchSize(g.outputTrans.size) 
}

func(g *UpdateCuda) BindParams(
        i backends.Tensor,
        e backends.Tensor,
        o backends.Tensor,
        alpha float64,
        beta float64,
        noOp bool) {
    base.Assert(i.Dtype() == g.dtype && e.Dtype() == g.dtype && o.Dtype() == g.dtype)
    g.lib.ScratchBufferInit()
    g.outputData = g.outputTrans.BindParams(o, alpha, beta, noOp)
    g.outputDataSize = o.Size()
    // SKIPPED: Stream as launchArgs[0]
    g.launchArgs = []acc.KernelArgument{1.0, 0.0, i, e, g.outputData, 0}
}

// SKIPPED: Arguments 'repeat' and 'unbind'
func(g *UpdateCuda) Execute() {
    kernel := g.GetKernel(g.dtype, g.rs, false, "update")
    // SKIPPED: stream argument in memset
    g.lib.MemsetD32Async(g.outputData, 0, g.outputDataSize)
    kernel.Launch(g.grid, g.block, 0, g.launchArgs, g.staticArgs)
    g.outputTrans.Execute()
/* SKIPPED
    if unbind {
        g.outputTrans.Unbind()
        g.outputData = nil
        g.launchArgs = nil
    }
*/
}

//
//    CompoundOps
//

// for kernels that can't compound these ops internally, use an external kernel
type CompoundOps struct {
    threads int
    dtype base.Dtype
    lib *CudaGenerator
    size int
    grid []int
    block []int
    n int
    args []acc.KernelArgument // SKIPPED: [0] = stream
    kernel *Kernel
}

func NewCompoundOps(lib *CudaGenerator, dtype base.Dtype, k int, n int) *CompoundOps {
    z := new(CompoundOps)
    z.Init(lib, dtype, k, n)
    return z
}

func(z *CompoundOps) Init(lib *CudaGenerator, dtype base.Dtype, k int, n int) {
    z.threads = 128
    for threads := 1024; threads >= 128; threads /= 2 {
        if threads * 8 <= n {
            z.threads = threads
            break
        }
    }
    z.dtype = dtype
    z.lib = lib
    z.size = k * n * dtype.ItemSize()
    z.grid = []int{k, 1, 1}
    z.block = []int{z.threads, 1, 1}
    z.n = n
}

func(z *CompoundOps) BindParams(
        o backends.Tensor,
        x backends.Tensor,
        bias backends.Tensor,
        bsum backends.Tensor,
        alpha float64,
        beta float64,
        relu bool,
        brelu bool) acc.DeviceAllocation {
    if bsum != nil || bias != nil || relu || brelu || beta != 0.0 || alpha != 1.0 {
        // beta is reused as slope param in Prelu / BpropPrelu
        doBeta := (beta != 0.0 || alpha != 1.0) && !relu && !brelu
        if x == nil {
            x = o
        }
        var inputData acc.DeviceAllocation
        if bias != nil || relu || brelu || beta != 0.0 || alpha != 1.0 {
            inputData = z.lib.ScratchBufferOffset(z.size)
        } else {
            inputData = o.(*acc.AccTensor).AccData()
        }
        // SKIPPED: stream as args[0]
        z.args = []acc.KernelArgument{
            o,
            bsum,
            bias,
            inputData,
            x,
            alpha,
            beta,
            z.n,
        }
        z.kernel = 
            getCompoundOpsKernel(
                z.lib,
                z.dtype, 
                z.threads, 
                bias!=nil, 
                bsum!=nil, 
                doBeta, 
                relu, 
                brelu)
        return inputData
    }

    z.kernel = nil
    return o.(*acc.AccTensor).AccData()
}

func(z *CompoundOps) Execute() {
    if z.kernel != nil {
        z.kernel.Launch(z.grid, z.block, 0, z.args, nil)
    }
}

// ACHTUNG: Is this method necessary in generator scenario?
func(z *CompoundOps) Unbind() {
    z.kernel = nil
    z.args = nil
}

var compoundOpsKernelCode = `
{{.common}}

#define THREADS {{.threads}}

__global__ void {{.name}}(
    {{.type}} *Out,
    float *Bsum,
    const float *__restrict__ Bias,
    const {{.type}} *__restrict__ In,
    const {{.type}} *__restrict__ X,
    float alpha, 
    float beta, 
    int N)
{
    const int tid  = threadIdx.x;
    const int bid  = blockIdx.x;

    In += bid*N + tid;
    {{.inits}}

    for (int i = tid; i < N; i += THREADS) {
        float data = {{.cvt_in}}(*In); 
        In += THREADS;
        {{.loads}}

        {{.ops}}
    }
    {{.bsum}}
}
`
// end compoundOpsKernelCode

var compoundOpsKernelBsum = `
    __shared__ float sPartials[THREADS];

    sPartials[tid] = sum;
    __syncthreads();

    #pragma unroll
    for (int a = THREADS >> 1; a > 32; a >>= 1) {
        if (tid < a)
            sPartials[tid] += sPartials[tid + a];
        __syncthreads();
    }
    if (tid < 32) {
        sum = sPartials[tid] + sPartials[tid + 32];

        #pragma unroll
        for (int i = 16; i > 0; i >>= 1)
            sum += __shfl_xor_sync(0xffffffff, sum, i);

        if (tid == 0)
            Bsum[bid] = sum;
    }
`
// end compoundOpsKernelBsum

func getCompoundOpsKernel(
        lib *CudaGenerator, 
        dtype base.Dtype, 
        threads int, 
        bias bool,
        bsum bool,
        beta bool,
        relu bool,
        brelu bool) *Kernel {
    key := 
        fmt.Sprintf("compound_ops_%s_%d_%t_%t_%t_%t_%t", 
            dtype, threads, bias, bsum, beta, relu, brelu)
    kernel := lib.LookupKernel(key)
    if kernel != nil {
        return kernel
    }
    kernel = getCompoundOpsKernelRaw(lib, dtype, threads, bias, bsum, beta, relu, brelu)
    lib.RegisterKernel(key, kernel)
    return kernel
}

func getCompoundOpsKernelRaw(
        lib *CudaGenerator, 
        dtype base.Dtype, 
        threads int, 
        bias bool,
        bsum bool,
        beta bool,
        relu bool,
        brelu bool) *Kernel {
    kernelName := "compound"
    if bias {
        kernelName += "_bias"
    }
    if relu {
        kernelName += "_relu"
    }
    if beta {
        kernelName += "_beta"
    }
    if brelu {
        kernelName += "_brelu"
    }
    if bsum {
        kernelName += "_bsum"
    }

    common := commonRound["nearest"][dtype]
    if dtype == base.Float16 {
        common += commonFp16toFp32
    }

    vals := map[string]interface{}{
        "name": kernelName,
        "threads": threads,
        "common": common,
        "type": ewTypes[dtype]["type"],
        "cvt_in": ewTypes[dtype]["cvt"],
        "cvt_out" : ewTypes[dtype]["cvt_out"],
        "bsum": "",
    }
    var inits []string
    if beta || brelu {
        inits = append(inits, "X += bid * N + tid;")
    }
    if bias {
        inits = append(inits, "Bias += bid;")
    }
    if bias || relu || beta || brelu {
        inits = append(inits, "Out += bid * N + tid;")
    }
    if bsum {
        inits = append(inits, "float sum = 0.0f;")
    }
    vals["inits"] = strings.Join(inits, "\n    ")

    var loads []string
    if beta || brelu {
        loads = append(loads, fmt.Sprintf("float x = %s(*X);  X += THREADS;", vals["cvt_in"]))
    }
    if bias {
        loads = append(loads, "float bias = *Bias;")
    }
    vals["loads"] = strings.Join(loads, "\n        ")

    var ops []string
    if bias {
        ops = append(ops, "data += bias;")
    }
    if relu {
        ops = append(ops, "data = max(data, 0.0f) + min(data, 0.0f) * beta;")
    }
    if beta {
        ops = append(ops, "data = data * alpha + x * beta;")
    }
    if brelu {
        ops = append(ops, "data *= (x > 0.0f) + (x < 0.0f) * beta;")
    }
    if bsum {
        ops = append(ops, "sum += data;")
    }
    if bias || relu || beta || brelu {
        ops = append(ops, fmt.Sprintf("*Out = %s(data); Out += THREADS;", vals["cvt_out"]))
    }
    vals["ops"] = strings.Join(ops, "\n        ")

    if bsum {
        vals["bsum"] = compoundOpsKernelBsum
    }

    code := executeTemplate(compoundOpsKernelCode, vals)

    return NewKernel(lib, kernelName, code)
}

//
//    FilterDimShuffle
//

type FilterDimShuffle struct {
    lib *CudaGenerator
    dim []int
    size int
    otype base.Dtype
    grid []int
    block []int
    launchArgs []acc.KernelArgument
    staticArgs []int
    kernel *Kernel
}

func NewFilterDimShuffle(
        lib *CudaGenerator,
        dtype base.Dtype,
        c int,
        t int,
        r int,
        s int,
        k int) *FilterDimShuffle {
    z := new(FilterDimShuffle)
    z.Init(lib, dtype, c, t, r, s, k)
    return z
}

func(z *FilterDimShuffle) Init(
        lib *CudaGenerator,
        dtype base.Dtype,
        c int,
        t int,
        r int,
        s int,
        k int) {
    gridC := ceilDiv(c, 32)
    gridK := ceilDiv(k, 32)
    z.lib = lib
    z.dim = []int{c, t, r, s, k}
    z.size = c * t * r * s * k * dtype.ItemSize()
    z.otype = dtype
    z.grid = []int{gridK, gridC, t*r*s}
    z.block = []int{32, 8, 1}
    sk := s * k
    rsk := r * s * k
    trsk := t * rsk
    sc := s * c
    rsc := r * sc
    trsc := t * rsc
    rs := r * s
    z.staticArgs =
        Flatten(
            trsk, 
            rsk, 
            sk, 
            k, 
            trsc, 
            rsc, 
            sc, 
            c,
            rs, 
            t, 
            r, 
            s, 
            magic32(t*rs, rs), 
            magic32(rs, s))
}

func(z *FilterDimShuffle) BindParams(f backends.Tensor) acc.DeviceAllocation {
    filterData := z.lib.ScratchBufferOffset(z.size)
    // SKIPPED: stream as launchArgs[0]
    z.launchArgs = []acc.KernelArgument{filterData, f}
    z.kernel = getShuffleKernel(z.lib, z.otype, f.Dtype())
    return filterData
}

func(z *FilterDimShuffle) Execute() {
    z.kernel.Launch(z.grid, z.block, 0, z.launchArgs, z.staticArgs)
}

// ACHTUNG: Is this method necessary in generator scenario?
func(z *FilterDimShuffle) Unbind() {
    z.kernel = nil
    z.launchArgs = nil
}

var shuffleKernel = `
{{.common}}

__global__ void {{.kernel_name}}(
    {{.otype}} *out, 
    const {{.itype}} *in,
    int TRSK, 
    int RSK, 
    int SK, 
    int K,
    int TRSC, 
    int RSC, 
    int SC, 
    int C,
    int RS, 
    int T, 
    int R, 
    int S,
    int magic_RS, 
    int shift_RS,
    int magic_S,  
    int shift_S)
{
    __shared__ {{.otype}} tile[32][33];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bk = blockIdx.x;
    int bc = blockIdx.y;
    int trs = blockIdx.z;

    int k = bk * 32 + tx;
    int c = bc * 32 + ty;

    int t = magic_RS * trs; 
    t >>= shift_RS;
    int rs = trs - t * RS;

    int r = magic_S * rs; 
    r >>= shift_S;
    int s = rs - r * S;

    for (int j = 0; j < 32; j += 8) {
        int cj = c + j;
        if (cj < C && k < K)
            tile[ty + j][tx] = {{.cvt_out}}(in[cj * TRSK + t * RSK + r * SK + s * K + k]);
    }
    __syncthreads();

    k = bk * 32 + ty;
    c = bc * 32 + tx;

    // Mirror RST
    s = S - s - 1;
    r = R - r - 1;
    t = T - t - 1;

    for (int i = 0; i < 32; i += 8) {
        int ki = k + i;
        if (ki < K && c < C)
            out[ki * TRSC + t * RSC + r * SC + s * C + c] = tile[tx][ty + i];
    }
}
`
// end shuffleKernel

func getShuffleKernel(lib *CudaGenerator, otype base.Dtype, itype base.Dtype) *Kernel {
    key := fmt.Sprintf("shuffle_%s_%s", otype, itype)
    kernel := lib.LookupKernel(key)
    if kernel != nil {
        return kernel
    }
    kernel = getShuffleKernelRaw(lib, otype, itype)
    lib.RegisterKernel(key, kernel)
    return kernel
}

func getShuffleKernelRaw(lib *CudaGenerator, otype base.Dtype, itype base.Dtype) *Kernel {
    // Allow fp32 in and fp16 out
    cvtOut := ""
    if otype != itype {
        cvtOut = ewTypes[otype]["cvt_out"]
    }

    kernelName := "filter_dimshuffle"
    data := map[string]interface{}{
        "common": commonRound["nearest"][otype],
        "itype": ewTypes[itype]["type"],
        "otype": ewTypes[otype]["type"],
        "cvt_out" : cvtOut,
        "kernel_name": kernelName,
    }

    code := executeTemplate(shuffleKernel, data)

    return NewKernel(lib, kernelName, code)
}

//
//    UpdateConvReduce
//

// fast axis=0 reduction kernel used for deterministic update
type UpdateConvReduce struct {
    lib *CudaGenerator
    mpq bool
    size int
    grid []int
    block []int
    launchArgs []acc.KernelArgument
    staticArgs []int
    kernel *Kernel
}

func NewUpdateConvReduce(lib *CudaGenerator, gridMpq int, crstk int) *UpdateConvReduce {
    z := new(UpdateConvReduce)
    z.Init(lib, gridMpq, crstk)
    return z
}

func(z *UpdateConvReduce) Init(lib *CudaGenerator, gridMpq int, crstk int) {
    blocks := ceilDiv(crstk, 32)
    pqcrstk := gridMpq * crstk
    z.lib = lib
    z.mpq = (gridMpq > 1)
    z.size = pqcrstk * 4
    z.grid = []int{blocks, 1, 1}
    z.block = []int{32, 1, 1}
    z.staticArgs = []int{crstk, pqcrstk}
}

func(z *UpdateConvReduce) BindParams(
        u backends.Tensor, 
        alpha float64, 
        beta float64, 
        noOp bool) acc.DeviceAllocation {
    if z.mpq || alpha != 1.0 || beta != 0.0 || u.Dtype() != base.Float32 {
        updateData := z.lib.ScratchBufferOffset(z.size)
        var outputData acc.DeviceAllocation
        if noOp {
            outputData = updateData
        } else {
            outputData = u.(*acc.AccTensor).AccData()
        }
        // SKIPPED: stream as launchArgs[0]
        z.launchArgs = []acc.KernelArgument{outputData, updateData, alpha, beta}
        z.kernel = getUpdateConvReduceKernel(z.lib, u.Dtype(), beta!=0.0)
        return updateData
    }
    z.kernel = nil
    if  noOp {
        return z.lib.ScratchBufferOffset(z.size)
    }
    return u.(*acc.AccTensor).AccData()
}

func(z *UpdateConvReduce) Execute() {
    if z.kernel != nil {
        z.kernel.Launch(z.grid, z.block, 0, z.launchArgs, z.staticArgs)
    }
}

// ACHTUNG: Is this method necessary in generator scenario?
func(z *UpdateConvReduce) Unbind() {
    z.kernel = nil
    z.launchArgs = nil
}

var updateConvReduceKernel = `
{{.common}}

__global__ void {{.kernel_name}}(
    {{.type}} *Out, 
    const float *In, 
    float alpha, 
    float beta, 
    int CRSTK, 
    int PQCRSTK)
{
    int offset = blockIdx.x * 32 + threadIdx.x;

    if (offset < CRSTK) {
        float sum = 0.0f;
        int i0 = offset;
        while (i0 < PQCRSTK) {
            int i1 = i0 + CRSTK;
            int i2 = i1 + CRSTK;
            int i3 = i2 + CRSTK;

            sum += In[i0];
            sum += i1 < PQCRSTK ? In[i1] : 0.0f;
            sum += i2 < PQCRSTK ? In[i2] : 0.0f;
            sum += i3 < PQCRSTK ? In[i3] : 0.0f;

            i0 = i3 + CRSTK;
        }
        Out[offset] = {{.output}};
    }
}
`
// end updateConvReduceKernel

func getUpdateConvReduceKernel(lib *CudaGenerator, dtype base.Dtype, beta bool) *Kernel {
    key := fmt.Sprintf("update_conv_reduce_%s_%t", dtype, beta)
    kernel := lib.LookupKernel(key)
    if kernel != nil {
        return kernel
    }
    kernel = getUpdateConvReduceKernelRaw(lib, dtype, beta)
    lib.RegisterKernel(key, kernel)
    return kernel
}

func getUpdateConvReduceKernelRaw(lib *CudaGenerator, dtype base.Dtype, beta bool) *Kernel {
    common := commonRound["nearest"][dtype]
    if dtype == base.Float16 {
        common += commonFp16toFp32
    }

    cvtIn := ewTypes[dtype]["cvt"]
    cvtOut := ewTypes[dtype]["cvt_out"]
    kernelName := "update_conv_reduce"

    templateVals := map[string]interface{}{
        "common": common,
        "type": ewTypes[dtype]["type"],
        "cvt_in": cvtIn,
        "cvt_out": cvtOut,
        "kernel_name": kernelName,
    }
    var output string
    if beta {
        output = fmt.Sprintf("%s(sum * alpha + %s(Out[offset]) * beta)", cvtOut, cvtIn)
    } else {
        output = fmt.Sprintf("%s(sum * alpha)", cvtOut)
    }
    templateVals["output"] = output

    code := executeTemplate(updateConvReduceKernel, templateVals)

    return NewKernel(lib, kernelName, code)
}

//
//    Kernel
//

// TODO: Relocate this? (- yes if Kernel can be used not only for convolutions)

type Kernel struct {
    lib *CudaGenerator
    name string
    code string
}

func NewKernel(lib *CudaGenerator, name string, code string) *Kernel {
    return &Kernel{lib, name, code}
}

func(k *Kernel) Launch(
        grid []int, 
        block []int, 
        shared int, 
        launchArgs []acc.KernelArgument, 
        staticArgs []int) {
    b := k.lib

    // staticArgs are passed as separate parameter to leave opportunity
    // for handling them in special way (e.g., as constants defined in kernel code)
    var args []acc.KernelArgument
    if staticArgs == nil {
        args = launchArgs
    } else {
        n1 := len(launchArgs)
        n2 := len(staticArgs)
        args = make([]acc.KernelArgument, n1+n2)
        copy(args, launchArgs)
        for i := 0; i < n2; i++ {
            args[n1+i] = staticArgs[i]
        }
    }

    id, ok := b.kernelIdMap[k]
    if !ok {
        id = b.makeKernelId()
        b.kernelIdMap[k] = id
        code := strings.Replace(k.code, k.name, id, 1)
        code = beautifyKernel(code, false)
        // device code for kernel
        b.EnterKernel()
        b.WriteLine("// %s", k.name)
        b.WriteChunk(code)
        b.ExitKernel()
    }

    sig := makeKernelSig(k.code, k.name)
    argList := makeKernelArgs(sig, args)

    // host code for kernel launch
    base.Assert(len(grid) == 3 && len(block) == 3)
    var strGrid, strBlock string
    if grid[1] != 1 || grid[2] != 1 || block[1] != 1 || block[2] != 1 {
        strGrid = fmt.Sprintf("dim3(%d, %d, %d)", grid[0], grid[1], grid[2])
        strBlock = fmt.Sprintf("dim3(%d, %d, %d)", block[0], block[1], block[2])
    } else {
        strGrid = fmt.Sprintf("%d", grid[0])
        strBlock = fmt.Sprintf("%d", block[0])
    }

    var launchConf string
    if shared != 0 {
        launchConf = fmt.Sprintf("%s, %s, %d", strGrid, strBlock, shared)
    } else {
        launchConf = fmt.Sprintf("%s, %s", strGrid, strBlock)
    }

    if len(args) <= 3 {
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
}

//
//    Local helpers
//

// flatten a nested list of lists or values
// only one level of nesting and int values supported here;
// this should be sufficient
func Flatten(lst ...interface{}) []int {
    var result []int
    for _, x := range lst {
        switch v := x.(type) {
        case int:
            result = append(result, v)
        case []int:
            for _, t := range v {
                result = append(result, t)
            }
        }
    }
    return result
}

// TODO: Verify correctness of the following code
//     Check whether any variables must be uint rather than int

func ceilDiv(x int, y int) int {
    // ACHTUNG: Works for positive x, y only
    base.Assert(x >= 0 && y > 0)
    return (x + y - 1) / y
}

func bitCount(n int) int {
    // TODO: Find idiomatic implementation
    return len(fmt.Sprintf("%b", n))
}

// Magic numbers and shift amounts for integer division
// Suitable for when nmax*magic fits in 32 bits
// Shamelessly pulled directly from:
// http://www.hackersdelight.org/hdcodetxt/magicgu.py.txt
func magic32(nmax int, d int) []int {
    nc := ((nmax + 1) / d) * d - 1
    nbits := bitCount(nmax)
    pstop := 2 * nbits + 1
    for p := 0; p < pstop; p++ {
        x := 1 << uint(p)
        y := d - 1 - (x - 1) % d
        if x > nc * y {
            m := (x + y) / d
            return []int{m, p}
        }
    }
    base.ValueError("Can't find magic number for division")
    return nil
}

// Magic numbers and shift amounts for integer division
// Suitable for when nmax*magic fits in 64 bits and the shift
// lops off the lower 32 bits
func magic64(d int) []int {
    // 3 is a special case that only ends up in the high bits
    // if the nmax is 0xffffffff
    // we can't use 0xffffffff for all cases as some return a 33 bit
    // magic number
    nmax := 0x7fffffff
    if d == 3 {
        nmax = 0xffffffff
    }
    v := magic32(nmax, d)
    magic, shift := v[0], v[1]
    if magic != 1 {
        shift -= 32
    }
    return []int{magic, shift}
}

