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

package main

import (
    "fmt"
    "fragata/arhat/backends"
    "fragata/arhat/base"
    "fragata/arhat/generators/cpu"
    "fragata/arhat/generators/cuda"
    "math"
    "os"
    "path/filepath"
    "strings"
)

//
//    Configurable options
//

var (
    cpuDtype = base.Float32 // originaly Float64
    computeCapability = [2]int{6, 0}
    ones = false
    outDir = "output"
    emulateDirect = true   // use CUDA instead of MaxAs Direct kernels
    emulateWinograd = true // use CUDA instead of MaxAs Winograd kernels
    splitTests = false     // split tests for CUDA devices with less than 8 GB memory
)

//
//    Device options
//

type Opts map[string]interface{}

func(o Opts) String() string {
    s := "{"
    for k, v := range o {
        if len(s) != 1 {
            s += ", "
        }
        var r string
        switch x := v.(type) {
        case bool:
            r = fmt.Sprintf("%t", x)
        case float64:
            r = fmt.Sprintf("%g", x)
        default:
            // cannot happen
            r = "?"
        }
        s += fmt.Sprintf("\"%s\": %s", k, r)
    }
    s += "}"
    return s
}

func(o Opts) Clone() Opts {
    r := make(Opts)
    for k, v := range o {
        r[k] = v
    }
    return r
}

func(o Opts) GetBool(key string) bool {
    val, ok := o[key]
    if !ok {
        return false
    }
    return val.(bool)
}

func(o Opts) GetFloat(key string, defVal float64) float64 {
    val, ok := o[key]
    if !ok {
        return defVal
    }
    return val.(float64)
}

func(o Opts) GetTensor(key string) backends.Tensor {
    val, ok := o[key]
    if !ok {
        return nil
    }
    if val == nil {
        return nil
    }
    return val.(backends.Tensor)
}

//
//    Device kernel wrappers
//

type Kernel interface {
    BindParams(
        devI1 backends.Tensor, 
        devI2 backends.Tensor, 
        devO backends.Tensor, 
        devOpts Opts)
    Execute()
}

// FpropCudaKernel

type FpropCudaKernel struct {
    kernels *cuda.FpropCuda
}

func NewFpropCudaKernel(
        lib *cuda.CudaGenerator, 
        dtype base.Dtype, 
        params *cuda.ConvParams) *FpropCudaKernel {
    kernels := cuda.NewFpropCuda(lib, dtype, params)
    return &FpropCudaKernel{kernels}
}

func(k *FpropCudaKernel) BindParams(
        devI1 backends.Tensor, 
        devI2 backends.Tensor, 
        devO backends.Tensor, 
        devOpts Opts) {
    x := devOpts.GetTensor("X")
    bias := devOpts.GetTensor("bias")
    bsum := devOpts.GetTensor("bsum")
    alpha := devOpts.GetFloat("alpha", 1.0)
    beta := devOpts.GetFloat("beta", 0.0)
    relu := devOpts.GetBool("relu")
    brelu := devOpts.GetBool("brelu")
    slope := devOpts.GetFloat("slope", 0.0)
    k.kernels.BindParams(
        devI1,
        devI2,
        devO,
        x,
        bias,
        bsum,
        alpha,
        beta,
        relu,
        brelu,
        slope)
}

func(k *FpropCudaKernel) Execute() {
    k.kernels.Execute()
}

// BpropCudaKernel

type BpropCudaKernel struct {
    kernels *cuda.BpropCuda
}

func NewBpropCudaKernel(
        lib *cuda.CudaGenerator, 
        dtype base.Dtype, 
        params *cuda.ConvParams) *BpropCudaKernel {
    kernels := cuda.NewBpropCuda(lib, dtype, params)
    return &BpropCudaKernel{kernels}
}

func(k *BpropCudaKernel) BindParams(
        devI1 backends.Tensor, 
        devI2 backends.Tensor, 
        devO backends.Tensor, 
        devOpts Opts) {
    x := devOpts.GetTensor("X")
    bias := devOpts.GetTensor("bias")
    bsum := devOpts.GetTensor("bsum")
    alpha := devOpts.GetFloat("alpha", 1.0)
    beta := devOpts.GetFloat("beta", 0.0)
    relu := devOpts.GetBool("relu")
    brelu := devOpts.GetBool("brelu")
    slope := devOpts.GetFloat("slope", 0.0)
    k.kernels.BindParams(
        devI1,
        devI2,
        devO,
        x,
        bias,
        bsum,
        alpha,
        beta,
        relu,
        brelu,
        slope)
}

func(k *BpropCudaKernel) Execute() {
    k.kernels.Execute()
}

// UpdateCudaKernel

type UpdateCudaKernel struct {
    kernels *cuda.UpdateCuda
}

func NewUpdateCudaKernel(
        lib *cuda.CudaGenerator, 
        dtype base.Dtype, 
        params *cuda.ConvParams) *UpdateCudaKernel {
    kernels := cuda.NewUpdateCuda(lib, dtype, params)
    return &UpdateCudaKernel{kernels}
}

func(k *UpdateCudaKernel) BindParams(
        devI1 backends.Tensor, 
        devI2 backends.Tensor, 
        devO backends.Tensor, 
        devOpts Opts) {
    alpha := devOpts.GetFloat("alpha", 1.0)
    beta := devOpts.GetFloat("beta", 0.0)
    k.kernels.BindParams(devI1, devI2, devO, alpha, beta, false)
}

func(k *UpdateCudaKernel) Execute() {
    k.kernels.Execute()
}

// kernel factories

func FpropCuda(
        lib *cuda.CudaGenerator, 
        dtype base.Dtype, 
        params *cuda.ConvParams,
        override bool) Kernel {
    return NewFpropCudaKernel(lib, dtype, params)
}

func BpropCuda(
        lib *cuda.CudaGenerator, 
        dtype base.Dtype, 
        params *cuda.ConvParams,
        override bool) Kernel {
    return NewBpropCudaKernel(lib, dtype, params)
}

func UpdateCuda(
        lib *cuda.CudaGenerator, 
        dtype base.Dtype, 
        params *cuda.ConvParams,
        override bool) Kernel {
    return NewUpdateCudaKernel(lib, dtype, params)
}

//
//    CPU kernel wrappers
//

type CpuKernel func(
        layer backends.ConvLayer,
        cpuI1 backends.Tensor, 
        cpuI2 backends.Tensor, 
        cpuO backends.Tensor, 
        cpuOpts Opts)

func FpropCpu(
        layer backends.ConvLayer,
        cpuI1 backends.Tensor, 
        cpuI2 backends.Tensor, 
        cpuO backends.Tensor, 
        cpuOpts Opts) {
    x := cpuOpts.GetTensor("X")
    bias := cpuOpts.GetTensor("bias")
    bsum := cpuOpts.GetTensor("bsum")
    alpha := cpuOpts.GetFloat("alpha", 1.0)
    beta := cpuOpts.GetFloat("beta", 0.0)
    relu := cpuOpts.GetBool("relu")
    brelu := cpuOpts.GetBool("brelu")
    slope := cpuOpts.GetFloat("slope", 0.0)
    cpuLayer := layer.(*cpu.ConvLayer)
    cpuLayer.FpropConv(
        cpuI1,
        cpuI2,
        cpuO,
        x,
        bias,
        bsum,
        alpha,
        beta,
        relu,
        brelu,
        slope)
}

func BpropCpu(
        layer backends.ConvLayer,
        cpuI1 backends.Tensor, 
        cpuI2 backends.Tensor, 
        cpuO backends.Tensor, 
        cpuOpts Opts) {
    x := cpuOpts.GetTensor("X")
    bias := cpuOpts.GetTensor("bias")
    bsum := cpuOpts.GetTensor("bsum")
    alpha := cpuOpts.GetFloat("alpha", 1.0)
    beta := cpuOpts.GetFloat("beta", 0.0)
    relu := cpuOpts.GetBool("relu")
    brelu := cpuOpts.GetBool("brelu")
    slope := cpuOpts.GetFloat("slope", 0.0)
    cpuLayer := layer.(*cpu.ConvLayer)
    cpuLayer.BpropConv(
        cpuI1,
        cpuI2,
        cpuO,
        x,
        bias,
        bsum,
        alpha,
        beta,
        relu,
        brelu,
        slope)
}

func UpdateCpu(
        layer backends.ConvLayer,
        cpuI1 backends.Tensor, 
        cpuI2 backends.Tensor, 
        cpuO backends.Tensor, 
        cpuOpts Opts) {
    alpha := cpuOpts.GetFloat("alpha", 1.0)
    beta := cpuOpts.GetFloat("beta", 0.0)
    cpuLayer := layer.(*cpu.ConvLayer)
    cpuLayer.UpdateConv(cpuI1, cpuI2, cpuO, alpha, beta, nil)
}

//
//    Configuration
//

var (
    //                   D,   H,     W,    T,   R,   S,  pad,      str
    conv_1x1     = []int{1,   14,    14,   1,   1,   1,  0, 0, 0,  1, 1, 1}
    conv_3x3     = []int{1,   14,    14,   1,   3,   3,  0, 1, 1,  1, 1, 1}
    conv_3x3p0   = []int{1,   14,    14,   1,   3,   3,  0, 0, 0,  1, 1, 1}
    conv_3x3p2   = []int{1,   14,    14,   1,   3,   3,  0, 2, 2,  1, 1, 1}
    conv_3x3s2   = []int{1,   14,    14,   1,   3,   3,  0, 1, 1,  1, 2, 2}
    conv_1x3     = []int{1,   14,    14,   1,   1,   3,  0, 0, 1,  1, 1, 1}
    conv_3x1     = []int{1,   14,    14,   1,   3,   1,  0, 1, 0,  1, 1, 1}
    conv_5x5     = []int{1,   14,    14,   1,   5,   5,  0, 2, 2,  1, 1, 1}
    conv_11x11s4 = []int{1,  224,   224,   1,  11,  11,  0, 2, 2,  1, 4, 4}
    conv_1x1x1   = []int{7,    7,     7,   1,   1,   1,  0, 0, 0,  1, 1, 1}
    conv_3x3x3   = []int{7,    7,     7,   3,   3,   3,  1, 1, 1,  1, 1, 1}
    conv_3x3x3s2 = []int{7,    7,     7,   3,   3,   3,  1, 1, 1,  2, 2, 2}
    conv_3x3L    = []int{1,  200,   200,   1,   3,   3,  0, 1, 1,  1, 1, 1}
    conv_1D      = []int{1,   13,  3263,   1,  13,  11,  0, 0, 0,  1, 1, 3}
)

type Config struct {
    kernel string
    n int
    c int
    k int
    determ bool
    cmpnd bool
    xtern bool
    conv [][]int
}

var configs1 = []Config{
    //     Kernel                    N    C    K   Determ Cmpnd  Xtern  conv
    Config{"FpropCuda",              32,  32,  32, true,  true,  false, [][]int{conv_3x3}},
    Config{"BpropCuda",              32,  32,  32, true,  true,  false, [][]int{conv_3x3}},
    Config{"UpdateCuda",             32,  32,  32, true,  true,  false, [][]int{conv_3x3}},
    Config{"FpropCuda",              32,  32,  32, true,  false, false, 
        [][]int{conv_1x1, conv_3x3s2, conv_1x3, conv_3x1, conv_5x5}},
    Config{"BpropCuda",              32,  32,  32, true,  false, false, 
        [][]int{conv_1x1, conv_3x3s2, conv_1x3, conv_3x1, conv_5x5}},
    Config{"UpdateCuda",             32,  32,  32, true,  false, false, 
        [][]int{conv_1x1, conv_3x3s2, conv_1x3, conv_3x1, conv_5x5}},
    Config{"FpropCuda",              32,   3,  64, true,  false, false, [][]int{conv_11x11s4}},
    Config{"UpdateCuda",             32,   3,  64, true,  false, false, [][]int{conv_11x11s4}},

    Config{"FpropDirect",            32,  32,  64, true,  true,  false, [][]int{conv_3x3, conv_3x3L}},
    Config{"BpropDirect",            32,  64,  32, true,  true,  false, [][]int{conv_3x3, conv_3x3L}},
    Config{"UpdateDirect",           32,  32,  32, true,  true,  false, [][]int{conv_3x3, conv_3x3L}},
    Config{"UpdateDirect",           32,  32,  32, false, true,  false, [][]int{conv_3x3, conv_3x3L}},

    Config{"FpropDirect",            32,  32,  64, true,  false, false, 
        [][]int{conv_1x1, conv_3x3s2, conv_1x3, conv_3x1, conv_5x5, conv_3x3x3, conv_1x1x1, conv_3x3x3s2}},
    Config{"BpropDirect",            32,  64,  32, true,  false, false, 
        [][]int{conv_1x1, conv_3x3s2, conv_1x3, conv_3x1, conv_5x5, conv_3x3x3, conv_1x1x1, conv_3x3x3s2}},
    Config{"UpdateDirect",           32,  32,  32, true,  false, false, 
        [][]int{conv_1x1, conv_3x3s2, conv_1x3, conv_3x1, conv_5x5, conv_3x3x3, conv_1x1x1, conv_3x3x3s2}},
}

var configs2 = []Config{
    Config{"FpropDirect",            32,   3,  64, true,  false, false, [][]int{conv_11x11s4}},
    Config{"UpdateDirect",           32,   3,  32, true,  false, false, [][]int{conv_11x11s4}},

    Config{"FpropDirect",            32,  64, 128, true,  true,  false, [][]int{conv_3x3}},
    Config{"FpropDirect",            32,  32,  63, true,  true,  false, [][]int{conv_3x3, conv_3x3L}},
    Config{"FpropDirect",            32,  32,   1, true,  true,  false, [][]int{conv_3x3, conv_3x3L}},

    Config{"FpropDirect",            16,  32,  64, true,  false, false, [][]int{conv_3x3}},
    Config{"FpropDirect",             8,  32,  64, true,  false, false, [][]int{conv_3x3}},
    Config{"FpropDirect",             4,  32,  64, true,  false, false, [][]int{conv_3x3}},
    Config{"FpropDirect",             2,  32,  64, true,  false, false, [][]int{conv_3x3}},
    Config{"FpropDirect",             1,  32,  64, true,  true,  false, [][]int{conv_3x3}},

    Config{"UpdateDirect",           16,  32,  63, true,  false, false, [][]int{conv_3x3}},
    Config{"UpdateDirect",            8,  32,  64, true,  false, false, [][]int{conv_3x3}},
    Config{"UpdateDirect",            4,  32, 128, true,  false, false, [][]int{conv_3x3}},

    Config{"FpropDirect",            32,   1, 512, true,  false, false, [][]int{conv_1D}},
    Config{"FpropDirect",            16,   1, 512, true,  false, false, [][]int{conv_1D}},
    Config{"FpropDirect",             8,   1, 512, true,  false, false, [][]int{conv_1D}},
    Config{"FpropDirect",             4,   1, 512, true,  false, false, [][]int{conv_1D}},
    Config{"FpropDirect",             2,   1, 512, true,  false, false, [][]int{conv_1D}},
    Config{"FpropDirect",             1,   1, 512, true,  false, false, [][]int{conv_1D}},

    Config{"UpdateDirect",           32,   1, 512, true,  false, false, [][]int{conv_1D}},
    Config{"UpdateDirect",           16,   1, 512, true,  false, false, [][]int{conv_1D}},
    Config{"UpdateDirect",            8,   1, 512, true,  false, false, [][]int{conv_1D}},
    Config{"UpdateDirect",            4,   1, 512, true,  false, false, [][]int{conv_1D}},

    //     Kernel                    N    C    K   Determ Cmpnd  Xtern  conv
    Config{"FpropDirect",            64,  32,  64, true,  true,  false, [][]int{conv_3x3}},
    Config{"FpropDirect",            64,  32, 128, true,  true,  false, [][]int{conv_3x3}},
    Config{"FpropDirect",           128,  32,  32, true,  true,  false, [][]int{conv_3x3}},
    Config{"FpropDirect",           128,  32,  64, true,  true,  false, [][]int{conv_3x3}},
    Config{"FpropDirect",           128,  32, 128, true,  true,  false, [][]int{conv_3x3}},
    Config{"BpropDirect",            64,  64,  32, true,  true,  false, [][]int{conv_3x3}},
    Config{"BpropDirect",            64, 128,  32, true,  true,  false, [][]int{conv_3x3}},
    Config{"BpropDirect",           128,  32,  32, true,  true,  false, [][]int{conv_3x3}},
    Config{"BpropDirect",           128,  64,  32, true,  true,  false, [][]int{conv_3x3}},
    Config{"BpropDirect",           128, 128,  32, true,  true,  false, [][]int{conv_3x3}},

    Config{"FpropDirect",            64,  32,  64, true,  false, false, 
        [][]int{conv_1x1, conv_3x3s2, conv_1x3, conv_3x1, conv_5x5, conv_3x3x3, conv_1x1x1, conv_3x3x3s2}},
    Config{"FpropDirect",            64,  32, 128, true,  false, false, 
        [][]int{conv_1x1, conv_3x3s2, conv_1x3, conv_3x1, conv_5x5, conv_3x3x3, conv_1x1x1, conv_3x3x3s2}},
    Config{"FpropDirect",           128,  32,  32, true,  false, false, 
        [][]int{conv_1x1, conv_3x3s2, conv_1x3, conv_3x1, conv_5x5, conv_3x3x3, conv_1x1x1, conv_3x3x3s2}},
    Config{"FpropDirect",           128,  32,  64, true,  false, false, 
        [][]int{conv_1x1, conv_3x3s2, conv_1x3, conv_3x1, conv_5x5, conv_3x3x3, conv_1x1x1, conv_3x3x3s2}},
    Config{"FpropDirect",           128,  32, 128, true,  false, false, 
        [][]int{conv_1x1, conv_3x3s2, conv_1x3, conv_3x1, conv_5x5, conv_3x3x3, conv_1x1x1, conv_3x3x3s2}},
    Config{"BpropDirect",            64,  64,  32, true,  false, false, 
        [][]int{conv_1x1, conv_3x3s2, conv_1x3, conv_3x1, conv_5x5, conv_3x3x3, conv_1x1x1, conv_3x3x3s2}},
    Config{"BpropDirect",            64, 128,  32, true,  false, false, 
        [][]int{conv_1x1, conv_3x3s2, conv_1x3, conv_3x1, conv_5x5, conv_3x3x3, conv_1x1x1, conv_3x3x3s2}},
    Config{"BpropDirect",           128,  32,  32, true,  false, false, 
        [][]int{conv_1x1, conv_3x3s2, conv_1x3, conv_3x1, conv_5x5, conv_3x3x3, conv_1x1x1, conv_3x3x3s2}},
    Config{"BpropDirect",           128,  64,  32, true,  false, false, 
        [][]int{conv_1x1, conv_3x3s2, conv_1x3, conv_3x1, conv_5x5, conv_3x3x3, conv_1x1x1, conv_3x3x3s2}},
    Config{"BpropDirect",           128, 128,  32, true,  false, false, 
        [][]int{conv_1x1, conv_3x3s2, conv_1x3, conv_3x1, conv_5x5, conv_3x3x3, conv_1x1x1, conv_3x3x3s2}},

    Config{"FpropDirect",            64,   3,  64, true,  false, false, [][]int{conv_11x11s4}},
    Config{"FpropDirect",            64,   3, 128, true,  false, false, [][]int{conv_11x11s4}},
    Config{"FpropDirect",           128,   3,  32, true,  false, false, [][]int{conv_11x11s4}},
    Config{"FpropDirect",           128,   3,  64, true,  false, false, [][]int{conv_11x11s4}},

    Config{"FpropDirect",            64,  33,  56, true,  true,  false, [][]int{conv_3x3s2}},
    Config{"FpropDirect",            64,  33, 120, true,  true,  false, [][]int{conv_3x3s2}},
    Config{"FpropDirect",           128,  33,  56, true,  true,  false, [][]int{conv_3x3s2}},
    Config{"FpropDirect",           128,  33, 120, true,  true,  false, [][]int{conv_3x3s2}},
    Config{"FpropDirect",           128,  33, 248, true,  true,  false, [][]int{conv_3x3s2}},
}

var configs3 = []Config{
    //     Kernel                    N    C    K   Determ Cmpnd  Xtern  conv
    Config{"FpropWinograd_2x2_3x3",  32,  32,  32, true,  true,  false, [][]int{conv_3x3, conv_3x3L}},
    Config{"FpropWinograd_2x2_3x3",  32,  32,  32, true,  true,  true,  [][]int{conv_3x3}},
    Config{"BpropWinograd_2x2_3x3",  32,  32,  32, true,  true,  false, [][]int{conv_3x3, conv_3x3L}},
    Config{"BpropWinograd_2x2_3x3",  32,  32,  32, true,  true,  true,  [][]int{conv_3x3}},
    Config{"UpdateWinograd_3x3_2x2", 32,  32,  32, true,  true,  false, [][]int{conv_3x3}},
    Config{"UpdateWinograd_3x3_2x2", 32,  32,  32, false, true,  false, [][]int{conv_3x3}},
    Config{"FpropWinograd_4x4_3x3",  32,  32,  32, true,  true,  false, [][]int{conv_3x3}},
    Config{"FpropWinograd_4x4_3x3",  32,  32,  32, true,  true,  true,  [][]int{conv_3x3, conv_3x3L}},
    Config{"BpropWinograd_4x4_3x3",  32,  32,  32, true,  true,  false, [][]int{conv_3x3}},
    Config{"BpropWinograd_4x4_3x3",  32,  32,  32, true,  true,  true,  [][]int{conv_3x3, conv_3x3L}},
    Config{"UpdateWinograd_3x3_4x4", 32,  32,  32, true,  true,  false, [][]int{conv_3x3}},
    Config{"UpdateWinograd_3x3_4x4", 32,  32,  32, false, true,  false, [][]int{conv_3x3}},

    Config{"FpropWinograd_2x2_3x3",  32,  32,  32, true,  false, true,  [][]int{conv_3x3p0, conv_3x3p2}},
    Config{"BpropWinograd_2x2_3x3",  32,  32,  32, true,  false, true,  [][]int{conv_3x3p0, conv_3x3p2}},
    Config{"UpdateWinograd_3x3_2x2", 32,  32,  32, true,  false, false, [][]int{conv_3x3p0, conv_3x3p2}},
    Config{"FpropWinograd_4x4_3x3",  32,  32,  32, true,  false, true,  [][]int{conv_3x3p0, conv_3x3p2}},
    Config{"BpropWinograd_4x4_3x3",  32,  32,  32, true,  false, true,  [][]int{conv_3x3p0, conv_3x3p2}},
    Config{"UpdateWinograd_3x3_4x4", 32,  32,  32, true,  false, false, [][]int{conv_3x3p0, conv_3x3p2}},

    Config{"FpropWinograd_2x2_3x3",   1,  63,  63, true,  false, true,  [][]int{conv_3x3, conv_3x3L}},
    Config{"BpropWinograd_2x2_3x3",   1,  63,  63, true,  false, true,  [][]int{conv_3x3, conv_3x3L}},
    Config{"UpdateWinograd_3x3_2x2",  1,  63,  63, true,  false, false, [][]int{conv_3x3}},
    Config{"FpropWinograd_4x4_3x3",   1,  63,  63, true,  false, true,  [][]int{conv_3x3, conv_3x3L}},
    Config{"BpropWinograd_4x4_3x3",   1,  63,  63, true,  false, true,  [][]int{conv_3x3, conv_3x3L}},
    Config{"UpdateWinograd_3x3_4x4",  1,  63,  63, true,  false, false, [][]int{conv_3x3}},

    Config{"FpropWinograd_2x2_5x5",  32,  32,  32, false, true,  false, [][]int{conv_5x5}},
    Config{"BpropWinograd_2x2_5x5",  32,  32,  32, false, true,  false, [][]int{conv_5x5}},

    Config{"FpropWinograd_2x2_5x5",  32,  64, 192, false, false, false, [][]int{conv_5x5}},
    Config{"BpropWinograd_2x2_5x5",  32,  64, 192, false, false, false, [][]int{conv_5x5}},
    Config{"FpropWinograd_2x2_5x5",  16,  64, 192, false, false, false, [][]int{conv_5x5}},
    Config{"FpropWinograd_2x2_5x5",   8,  64, 192, false, false, false, [][]int{conv_5x5}},
    Config{"FpropWinograd_2x2_5x5",   4,  64, 192, false, false, false, [][]int{conv_5x5}},
    Config{"FpropWinograd_2x2_5x5",   2,  64, 192, false, false, false, [][]int{conv_5x5}},
}

var (
    fpropOpts = []Opts{
        Opts{},
        Opts{"slope": 0.0, "relu": true},
        Opts{"slope": 0.1, "relu": true},
        Opts{"bias": true},
        Opts{"bias": true, "slope": 0.0, "relu": true},
        Opts{"bias": true, "slope": 0.1, "relu": true},
        Opts{"bsum": true},
    }
    bpropOpts = []Opts{
        Opts{},
        Opts{"X": true, "slope": 0.0, "brelu": true},
        Opts{"X": true, "slope": 0.1, "brelu": true},
        Opts{"X": true, "bsum": true, "slope": 0.0, "brelu": true},
        Opts{"X": true, "bsum": true, "slope": 0.1, "brelu": true},
        Opts{"X": true, "alpha": 2.0, "beta": 3.0},
        Opts{"alpha": 2.0, "beta": 3.0},
    }
    updateOpts = []Opts{
        Opts{"alpha": 2.0, "beta": 3.0},
        Opts{},
    }
)

type KernelClass func(*cuda.CudaGenerator, base.Dtype, *cuda.ConvParams, bool) Kernel

var kernels = map[string]KernelClass{
    // CUDA
    "FpropCuda": FpropCuda,
    "BpropCuda": BpropCuda,
    "UpdateCuda": UpdateCuda,
    // MaxAs
    "FpropDirect": nil,
    "BpropDirect": nil,
    "UpdateDirect": nil,
    "FpropWinograd_2x2_3x3": nil,
    "BpropWinograd_2x2_3x3": nil,
    "UpdateWinograd_3x3_2x2": nil,
    "FpropWinograd_4x4_3x3": nil,
    "BpropWinograd_4x4_3x3": nil,
    "UpdateWinograd_3x3_4x4": nil,
    "FpropWinograd_2x2_5x5": nil,
    "BpropWinograd_2x2_5x5": nil,
}

func init() {
    if emulateDirect {
        kernels["FpropDirect"] = FpropCuda
        kernels["BpropDirect"] = BpropCuda
        kernels["UpdateDirect"] = UpdateCuda
    }
    if emulateWinograd {
        kernels["FpropWinograd_2x2_3x3"] = FpropCuda
        kernels["BpropWinograd_2x2_3x3"] = BpropCuda
        kernels["UpdateWinograd_3x3_2x2"] = UpdateCuda
        kernels["FpropWinograd_4x4_3x3"] = FpropCuda
        kernels["BpropWinograd_4x4_3x3"] = BpropCuda
        kernels["UpdateWinograd_3x3_4x4"] = UpdateCuda
        kernels["FpropWinograd_2x2_5x5"] = FpropCuda
        kernels["BpropWinograd_2x2_5x5"] = BpropCuda
    }
}

var (
    fpropKernels = map[string]bool{
        "FpropCuda": true,  
        "FpropDirect": true,  
        "FpropWinograd_2x2_3x3": true,  
        "FpropWinograd_4x4_3x3": true, 
        "FpropWinograd_2x2_5x5": true,
    }
    bpropKernels = map[string]bool{
        "BpropCuda": true,  
        "BpropDirect": true,  
        "BpropWinograd_2x2_3x3": true,  
        "BpropWinograd_4x4_3x3": true, 
        "BpropWinograd_2x2_5x5": true,
    }
    updateKernels = map[string]bool{
        "UpdateCuda": true, 
        "UpdateDirect": true, 
        "UpdateWinograd_3x3_2x2": true, 
        "UpdateWinograd_3x3_4x4": true,
    }
)

//
//    Testing
//

func main() {
    // GPU memory use (FP32), Tesla K40 
    //     configs1 2.715 GB
    //     configs2 1.574 GB
    //     configs3 2.316 GB
    //     no split 6.420 GB
    // Split tests for CUDA devices with less than 8 GB
    if splitTests {
        Test(configs1, outDir+"1")
        Test(configs2, outDir+"2")
        Test(configs3, outDir+"3")
    } else {
        n1 := len(configs1)
        n2 := len(configs2)
        n3 := len(configs3)
        configs := make([]Config, n1+n2+n3)
        copy(configs, configs1)
        copy(configs[n1:], configs2)
        copy(configs[n1+n2:], configs3)
        Test(configs, outDir)
    }
}

func Test(configs []Config, outDir string) {
    ng := NewCudaGenerator()
    nc := NewCpuGenerator()

    var context TestContext
    context.Init(ng, nc)

    numTests := 0

    for _, config := range configs {
        kernelName := config.kernel
        kernelClass, ok := kernels[kernelName]
        if !ok {
            base.TypeError("Unknown kernel class: %s", kernelName)
        }
        if kernelClass == nil {
            // not supported
            continue
        }

        n := config.n
        c := config.c
        k := config.k
//        determ := config.determ
        compound := config.cmpnd
        override := config.xtern
        convs := config.conv

        for _, conv := range convs {
            params := NewConvParams(n, c, k, conv)

//            ng.SetDeterministic(determ)

            layer := nc.NewConvLayer(cpuDtype, params)

            cudaParams := NewCudaConvParams(params, layer)

            var dtypes []base.Dtype
            if kernelName == "FpropCuda" || 
                    kernelName == "BpropCuda" || 
                    kernelName == "UpdateCuda" {
                // FP16 has not been tested yet for CUDA kernels
                dtypes = []base.Dtype{base.Float32}
            } else {
                dtypes = []base.Dtype{base.Float32, base.Float16}
            }

            // patch for emulation of MaxAs with Cuda kernels
            // skip tests with unsupported parameters
            if (emulateDirect && strings.Contains(kernelName, "Direct")) ||
                    (emulateWinograd && strings.Contains(kernelName, "Winograd")) {
                if cudaParams.D != 1 || cudaParams.T != 1 {
                    fmt.Printf("SKIPPED: 3D Convolution not supported by CUDA C kernels\n")
                    continue
                }
                if cudaParams.N % 32 != 0 {
                    fmt.Printf("SKIPPED: N dim must be multiple of 32\n")
                    continue
                }
                if cudaParams.K % 4 != 0 {
                    // vecSize = 4 for float32
                    fmt.Printf("SKIPPED: K dim must be multiple of 4\n")
                    continue
                }
                dtypes = []base.Dtype{base.Float32}
            }

            for _, dtype := range dtypes {
                ng.ScratchBufferReset()

                kernel := kernelClass(ng, dtype, cudaParams, override)

                var dimI1 []int
                var dimI2 []int
                var dimO []int
                var opts []Opts
                var cpuKernel CpuKernel
                back := false
                update := false
                switch {
                case fpropKernels[kernelName]:
                    dimI1 = layer.DimI()
                    dimI2 = layer.DimF()
                    dimO = layer.DimO()
                    opts = fpropOpts
                    cpuKernel = FpropCpu
                case bpropKernels[kernelName]:
                    dimI1 = layer.DimO()
                    dimI2 = layer.DimF()
                    dimO = layer.DimI()
                    opts = bpropOpts
                    cpuKernel = BpropCpu
                    back = true
                case updateKernels[kernelName]:
                    dimI1 = layer.DimI()
                    dimI2 = layer.DimO()
                    dimO = layer.DimF()
                    opts = updateOpts
                    cpuKernel = UpdateCpu
                    update = true
                default:
                    base.TypeError("Unknown kernel class: %s", kernelName)
                }

                if !compound {
                    opts = []Opts{Opts{}}
                }

                devI1 := ng.NewTensor(dimI1, dtype)
                devI2 := ng.NewTensor(dimI2, dtype)
                devO := ng.NewTensor(dimO, dtype)

                cpuI1 := nc.NewTensor(dimI1, cpuDtype)
                cpuI2 := nc.NewTensor(dimI2, cpuDtype)
                cpuO := nc.NewTensor(dimO, cpuDtype)

                var devB backends.Tensor
                var devS backends.Tensor
                var devX backends.Tensor

                var cpuB backends.Tensor
                var cpuS backends.Tensor
                var cpuX backends.Tensor

                if compound && !update {
                    devB = ng.NewTensor([]int{dimO[0], 1}, base.Float32)
                    devS = ng.NewTensor([]int{dimO[0], 1}, base.Float32)
                    cpuB = nc.NewTensor([]int{dimO[0], 1}, cpuDtype)
                    cpuS = nc.NewTensor([]int{dimO[0], 1}, cpuDtype)
                }

                if compound && back {
                    devX = ng.NewTensor(dimO, base.Float32)
                    cpuX = nc.NewTensor(dimO, cpuDtype)
                }

                for _, opt := range opts {
                    devOpts := opt.Clone()
                    cpuOpts := opt.Clone()

                    if opt.GetBool("bias") {
                        devOpts["bias"] = devB
                        cpuOpts["bias"] = cpuB
                    } else {
                        devOpts["bias"] = nil
                        cpuOpts["bias"] = nil
                    }

                    var devZ backends.Tensor
                    var cpuZ backends.Tensor
                    sizeZ := 0

                    if opt.GetBool("bsum") {
                        devOpts["bsum"] = devS
                        cpuOpts["bsum"] = cpuS
                        devZ = devS
                        cpuZ = cpuS
                        sizeZ = dimO[0]
                    } else {
                        devOpts["bsum"] = nil
                        cpuOpts["bsum"] = nil
                    }

                    if opt.GetBool("X") {
                        devOpts["X"] = devX
                        cpuOpts["X"] = cpuX
                    } else {
                        devOpts["X"] = nil
                        cpuOpts["X"] = nil
                    }

                    context.StartTest(
                        kernelName, 
                        conv, 
                        dtype, 
                        opt, 
                        devO.Size(), 
                        sizeZ)

                    offset := 0

                    context.InitCudaTensor(devI1, offset)
                    context.InitCpuTensor(cpuI1, offset)
                    offset += devI1.Size()

                    context.InitCudaTensor(devI2, offset)
                    context.InitCpuTensor(cpuI2, offset)
                    offset += devI2.Size()

                    context.InitCudaTensor(devO, offset)
                    context.InitCpuTensor(cpuO, offset)
                    offset += devO.Size()

                    if devB != nil {
                        context.InitCudaTensor(devB, offset)
                        context.InitCpuTensor(cpuB, offset)
                        offset += devB.Size()
                    }

                    if devS != nil {
                        context.InitCudaTensor(devS, offset)
                        context.InitCpuTensor(cpuS, offset)
                        offset += devS.Size()
                    }

                    if devX != nil {
                        context.InitCudaTensor(devX, offset)
                        context.InitCpuTensor(cpuX, offset)
                        offset += devX.Size()
                    }

                    kernel.BindParams(devI1, devI2, devO, devOpts)
                    kernel.Execute()

                    cpuKernel(layer, cpuI1, cpuI2, cpuO, cpuOpts)

                    context.EndTest(devO, cpuO, devZ, cpuZ, offset)

                    numTests++
                } // opt

                // ACHTUNG: Cannot do now; would be nice to have in the future:
                //     release devI1, devI2, devO, devB, devS, devX
                //     release cpuI1, cpuI2, cpuO, cpuB, cpuS, cpuX

            } // dtype
        } // conv
    } // config

    if numTests > 0 {
        context.OutputCode(outDir)
    } else {
        fmt.Printf("No valid tests for %s\n", outDir)
    }
}

func NewCudaGenerator() *cuda.CudaGenerator {
    ng := cuda.NewCudaGenerator(
        base.IntNone,            // rngSeed
        base.DtypeNone,          // defaultDtype
        base.IntNone,            // stochasticRound
        base.IntNone,            // deviceId
        computeCapability,       // computeCapability
        false,                   // bench
        base.IntNone,            // scratchSize
        base.IntNone,            // histBins
        base.IntNone,            // histOffset
        backends.CompatModeNone) // compatMode

    ng.ConfigureCodeOutput(false, "cuda_", "ncuda", "Cuda_")

    backends.SetBe(ng)

    return ng
}

func NewCpuGenerator() *cpu.CpuGenerator {
    nc := cpu.NewCpuGenerator(
        base.IntNone,            // rngSeed
        base.DtypeNone,          // defaultDtype
        base.IntNone,            // stochasticRound
        false,                   // bench
        base.IntNone,            // scratchSize
        base.IntNone,            // histBins
        base.IntNone,            // histOffset
        backends.CompatModeNone) // compatMode

    nc.ConfigureCodeOutput(false, "cpu_", "ncpu", "Cpu_")

    return nc
}

func NewConvParams(n int, c int, k  int, conv []int) *backends.ConvParams {
    // no suitable constructor in fragata/arhat/backends
    a := new(backends.ConvParams)
    a.Init()
    a.N = n
    a.C = c
    a.K = k
    a.D = conv[0]
    a.H = conv[1]
    a.W = conv[2]
    a.T = conv[3]
    a.R = conv[4]
    a.S = conv[5]
    a.PadD = conv[6]
    a.PadH = conv[7]
    a.PadW = conv[8]
    a.StrD = conv[9]
    a.StrH = conv[10]
    a.StrW = conv[11]
    return a
}

func NewCudaConvParams(
        params *backends.ConvParams, layer backends.ConvLayer) *cuda.ConvParams {
    // no suitable constructor in fragata/arhat/generators/cuda
    a := new(cuda.ConvParams)
    a.InitConv(params)
    mpq := layer.MPQ()
    a.M = mpq[0]
    a.P = mpq[1]
    a.Q = mpq[2]
    return a
}

//
//    TestContext
//

type TestDesc struct {
    info string
    sizeO int
    sizeS int
}

type TestContext struct {
    ng *cuda.CudaGenerator
    nc *cpu.CpuGenerator
    valsSize int
    descMap map[int]*TestDesc
    nextTestId int
}

func(c *TestContext) Init(ng *cuda.CudaGenerator, nc *cpu.CpuGenerator) {
    c.ng = ng
    c.nc = nc
    c.valsSize = 0
    c.descMap = make(map[int]*TestDesc)
    c.nextTestId = 1
}

func(c *TestContext) StartCode() {
    c.ng.StartCode()
    c.nc.StartCode()
}

func(c *TestContext) StartTest(
        kernelName string, 
        conv []int, 
        dtype base.Dtype, 
        opt Opts, 
        sizeO int, 
        sizeS int) {
    testId := c.nextTestId
    c.nextTestId++

    info := fmt.Sprintf("%s %s %s %s", kernelName, formatShape(conv), dtype, opt)
    c.descMap[testId] = &TestDesc{info: info, sizeO: sizeO, sizeS: sizeS}

    ng := c.ng
    ng.WriteLine("// %s", info)
    ng.WriteLine("void Test%02d(float *vals, void *&out, void *&sum) {", testId)
    ng.Indent(1)

    nc := c.nc
    nc.WriteLine("// %s", info)
    nc.WriteLine("void Test%02d(float *vals, void *&out, void *&sum) {", testId)
    nc.Indent(1)
}

func(c *TestContext) EndTest(
        devO backends.Tensor, 
        cpuO backends.Tensor,
        devS backends.Tensor,
        cpuS backends.Tensor,
        valsSize int) {
    ng := c.ng
    ng.WriteLine("out = %s;", ng.FormatBufferRef(devO, false))
    if devS != nil {
        ng.WriteLine("sum = %s;", ng.FormatBufferRef(devS, false))
    } else {
        ng.WriteLine("sum = nullptr;")
    }
    ng.Indent(-1)
    ng.WriteLine("}")
    ng.WriteLine("")

    nc := c.nc
    nc.WriteLine("out = %s;", nc.FormatBufferRef(cpuO, false))
    if cpuS != nil {
        nc.WriteLine("sum = %s;", nc.FormatBufferRef(cpuS, false))
    } else {
        nc.WriteLine("sum = nullptr;")
    }
    nc.Indent(-1)
    nc.WriteLine("}")
    nc.WriteLine("")

    if valsSize > c.valsSize {
        c.valsSize = valsSize
    }
}

func(c *TestContext) InitCudaTensor(tensor backends.Tensor, offset int) {
    // ACHTUNG: Float32 only
    ng := c.ng
    ng.WriteLine("CudaMemcpyHtodAsync(%s, &vals[%d], %d);", 
        ng.FormatBufferRef(tensor, false), offset, 4*tensor.Size());
}

func(c *TestContext) InitCpuTensor(tensor backends.Tensor, offset int) {
    nc := c.nc
    nc.WriteLine("CpuMemcpy(%s, &vals[%d], %d);", 
        nc.FormatBufferRef(tensor, false), offset, 4*tensor.Size());
}

func(c *TestContext) OutputCode(outDir string) {
    err := os.MkdirAll(outDir, 0777)
    if err != nil {
        base.RuntimeError("%s\n", err.Error())
    }

    c.ng.EndCode()
    err = c.ng.OutputCode(outDir)
    if err != nil {
        fmt.Printf("%s\n", err.Error())
        return
    }

    c.nc.EndCode()
    err = c.nc.OutputCode(outDir)
    if err != nil {
        fmt.Printf("%s\n", err.Error())
        return
    }

    err = c.OutputMain(outDir);
    if err != nil {
        base.RuntimeError("%s\n", err.Error())
    }
}

func(c *TestContext) OutputMain(outDir string) error {
    fn := filepath.Join(outDir, "main.cpp")
    fp, err := os.Create(fn)
    if err != nil {
        return err
    } 

    fmt.Fprintf(fp, "\n")
    fmt.Fprintf(fp, "#include <cmath>\n")
    fmt.Fprintf(fp, "#include \"cuda_host.h\"\n")
    fmt.Fprintf(fp, "#include \"cpu_host.h\"\n")
    fmt.Fprintf(fp, "\n")

    fmt.Fprintf(fp, "namespace ncuda {\n")
    fmt.Fprintf(fp, "    void Prolog();\n")
    for i := 1; i < c.nextTestId; i++ {
        fmt.Fprintf(fp, "    void Test%02d(float *vals, void *&out, void *&sum);\n", i)
    }
    fmt.Fprintf(fp, "} // ncuda\n")
    fmt.Fprintf(fp, "\n")

    fmt.Fprintf(fp, "namespace ncpu {\n")
    fmt.Fprintf(fp, "    void Prolog();\n")
    for i := 1; i < c.nextTestId; i++ {
        fmt.Fprintf(fp, "    void Test%02d(float *vals, void *&out, void *&sum);\n", i)
    }
    fmt.Fprintf(fp, "} // ncpu\n")
    fmt.Fprintf(fp, "\n")

    fmt.Fprintf(fp, "%s\n", codeVerify)

    fmt.Fprintf(fp, "void Main() {\n")
    fmt.Fprintf(fp, "    ncuda::Prolog();\n")
    fmt.Fprintf(fp, "    ncpu::Prolog();\n")
    fmt.Fprintf(fp, "\n")

    fmt.Fprintf(fp, "    size_t remain, total;\n")
    fmt.Fprintf(fp, "    arhat::cuda::CudaMemGetInfo(remain, total);\n")
    fmt.Fprintf(fp, "    double gb = 1024.0 * 1024.0 * 1024.0;\n")
    fmt.Fprintf(fp, "    %s\n",
        `printf("%.3fGB of %.3fGB Allocated (%.3fGB Remaining)\n", `+
        `(total - remain) / gb, total / gb, remain / gb);`)
    fmt.Fprintf(fp, "\n")

    fmt.Fprintf(fp, "    void *devO = nullptr;\n")
    fmt.Fprintf(fp, "    void *cpuO = nullptr;\n")
    fmt.Fprintf(fp, "    void *devS = nullptr;\n")
    fmt.Fprintf(fp, "    void *cpuS = nullptr;\n")
    fmt.Fprintf(fp, "\n")

    fmt.Fprintf(fp, "    float *vals = new float[%d];\n", c.valsSize)
    if ones {
        oneBits := math.Float32bits(1.0)
        fmt.Fprintf(fp, "    arhat::cpu::CpuMemsetD32(vals, %du, %d);\n", oneBits, c.valsSize)
    } else {
        fmt.Fprintf(fp, "    arhat::cpu::RngUniform(vals, 0.0, 1.0, %d);\n", c.valsSize)
    }
    fmt.Fprintf(fp, "\n")

    for i := 1; i < c.nextTestId; i++ {
        desc := c.descMap[i]
        info := strings.Replace(desc.info, "\"", "\\\"", -1)
        sizeO := desc.sizeO
        sizeS := desc.sizeS
        fmt.Fprintf(fp, "    printf(\"%s\\n\");\n", info)
        fmt.Fprintf(fp, "    ncuda::Test%02d(vals, devO, devS);\n", i)
        fmt.Fprintf(fp, "    ncpu::Test%02d(vals, cpuO, cpuS);\n", i)
        fmt.Fprintf(fp, "    Verify(devO, cpuO, %d, devS, cpuS, %d);\n", sizeO, sizeS)
        fmt.Fprintf(fp, "\n")
    }

    fmt.Fprintf(fp, "    delete[] vals;\n")
    fmt.Fprintf(fp, "}\n")
    fmt.Fprintf(fp, "\n")

    fmt.Fprintf(fp, "int main() {\n")
    fmt.Fprintf(fp, "    Main();\n")
    fmt.Fprintf(fp, "    return 0;\n")
    fmt.Fprintf(fp, "}\n")
    fmt.Fprintf(fp, "\n")

    fp.Close()

    return nil
}

func formatShape(shape []int) string {
    result := "["
    for i, s := range shape {
        if i != 0 {
            result += ", "
        }
        result += fmt.Sprintf("%d", s)
    }
    result += "]"
    return result
}

// templates for static functions

var codeVerify =
`void Verify(void *devO, void *cpuO, int sizeO, void *devS, void *cpuS, int sizeS) {
    float *cpuA = (float *)cpuO;
    float *devA = new float[sizeO];
    arhat::cuda::CudaMemcpyDtoh(devA, devO, sizeO * sizeof(float));
    float maxval = 0.0f;
    float maxdif = 0.0f;
    for (int i = 0; i < sizeO; i++) {
        float val = abs(cpuA[i]);
        if (val > maxval) {
            maxval = val;
        }
        float dif = abs(cpuA[i] - devA[i]);
        if (dif > maxdif) {
            maxdif = dif;
        }
    }
    float ratio = maxdif / maxval;
    delete[] devA;
    devA = nullptr;

    float ratio2 = 0.0f;
    if (sizeS != 0) {
        cpuA = (float *)cpuS;
        devA = new float[sizeS];
        arhat::cuda::CudaMemcpyDtoh(devA, devS, sizeS * sizeof(float));
        maxval = 0.0f;
        maxdif = 0.0f;
        for (int i = 0; i < sizeS; i++) {
            float val = abs(cpuA[i]);
            if (val > maxval) {
                maxval = val;
            }
            float dif = abs(cpuA[i] - devA[i]);
            if (dif > maxdif) {
                maxdif = dif;
            }
        }
        ratio2 = maxdif / maxval;
        delete[] devA;
    }

    const char *result = "OK";
    if (ratio > 0.01 || ratio2 > 0.01) {
        result = "FAIL";
    }
    printf("%s: RATIO %g RATIO2 %g\n", result, ratio, ratio2);
}
`

