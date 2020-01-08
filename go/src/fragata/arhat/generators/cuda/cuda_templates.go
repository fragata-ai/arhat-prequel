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
    "fragata/arhat/backends"
    kernels "fragata/arhat/kernels/cuda"
)

// names imported from kernels/cuda

var (
    // data
    initRandFunc = kernels.InitRandFunc
    initRandRoundFunc = kernels.InitRandRoundFunc
    finishRandFunc = kernels.FinishRandFunc
    commonKepler = kernels.CommonKepler
    commonUrandGen = kernels.CommonUrandGen
    commonFrand = kernels.CommonFrand
    commonRound = kernels.CommonRound
    commonFp16toFp32 = kernels.CommonFp16toFp32
    commonMaxAbs = kernels.CommonMaxAbs
    ewTypes = kernels.EwTypes
    ewStringsRound = kernels.EwStringsRound
    // functions
    format = kernels.Format
)

var ewTemplate = `
#define FLT_MAX 3.402823466E+38F
#define RAND_POOL_SIZE 65536

{{.common}}

#define THREADS {{.threads}}

__global__ void {{.name}}(
    unsigned* rand_state,
    {{.arguments}})
{
    const int tid  = threadIdx.x;
    const int bid  = blockIdx.x;

    extern __shared__ float sPartials[];

    {{.inits}}
` 

var stageTemplate = map[string]string{
    "loop": `

    for (int i = tid; i < n$0; i += THREADS) {
        {{.loads$0}}

        {{.ops$0}}
    }
`,

    "red32": `

    #pragma unroll
    for (int i = 16; i > 0; i >>= 1) {
        {{.shfl_red$0}}
    }

`,

    "red": `

    sPartials[tid] = {{.var_red$0}};
    __syncthreads();

    #pragma unroll
    for (int a = THREADS >> 1; a > 32; a >>= 1) {
        if (tid < a)
            {{.share1_red$0}}
        __syncthreads();
    }

    if (tid < 32) {
        {{.share2_red$0}}

        // __syncthreads(); // Seems to prevent a race condition but causes other problems

        #pragma unroll
        for (int i = 16; i > 0; i >>= 1)
            {{.shfl_red$0}}

        sPartials[tid] = {{.var_red$0}};
    }
    __syncthreads();
    {{.var_red$0}} = sPartials[0];
`,

    "red_ops": `

        {{.ops$0}}
`,

    "red_out": `

    if (tid == 0) {
        {{.ops$0}}
    }
`,
} 

var finTemplate = `
    {{.finish}}
}
`

var ewStrings = map[string]map[string]string{
    // 0: argId, 1: stage, 2: type, 3: cvt
    "in0": map[string]string{
        "arguments": "const $2* a$0_in, int row_strd$0, int col_strd$0",
        "inits": 
            "const $2* a$0_in$1 = a$0_in + bid * row_strd$0 + tid * col_strd$0;\n" +
            "int a$0_inc$1 = THREADS * col_strd$0;",
        "loads": 
            "float a$0 = $3(__ldg(a$0_in$1));\n" +
            "a$0_in$1 += a$0_inc$1;",
    },
    "in1": map[string]string{
        "arguments": "const $2* a$0_in, int row_strd$0, int col_strd$0, const int* take$0_in",
        "inits": 
            "const $2* a$0_in$1 = a$0_in + __ldg(take$0_in + bid) * row_strd$0 " +
                "+ tid * col_strd$0;\n" +
            "int a$0_inc$1 = THREADS * col_strd$0;",
        "loads": 
            "float a$0 = $3(__ldg(a$0_in$1));\n" +
            "a$0_in$1 += a$0_inc$1;",
    },
    "in2": map[string]string{
        "arguments": "const $2* a$0_in, int row_strd$0, int col_strd$0, const int* take$0_in",
        "inits": 
            "const $2* a$0_in$1 = a$0_in + bid * row_strd$0;\n" +
            "const int* take$0_in$1 = take$0_in + tid;",
        "loads": 
            "float a$0 = $3(__ldg(a$0_in$1 + __ldg(take$0_in$1)));\n" +
            "take$0_in$1 += THREADS;",
    },
    "out0": map[string]string{
        "arguments": "$2* a_out, int row_strd, int col_strd",
        "inits": 
            "a_out += bid * row_strd + tid * col_strd;\n" +
            "int out_inc = THREADS * col_strd;",
        "output": 
            "*a_out = $0;\n" +
            "a_out += out_inc;",
    },
    "out1": map[string]string{
        "arguments": "$2* a_out, int row_strd, int col_strd, const int* take_out",
        "inits": 
            "a_out += __ldg(take_out + bid) * row_strd + tid * col_strd;\n" +
            "int out_inc = THREADS * col_strd;",
        "output": 
            "*a_out = $0;\n" +
            "a_out += out_inc;",
    },
    "out2": map[string]string{
        "arguments": "$2* a_out, int row_strd, int col_strd, const int* take_out",
        "inits": 
            "a_out += bid * row_strd;\n" +
            "take_out += tid;",
        "output": 
            "*(a_out + __ldg(take_out)) = $0;\n" +
            "take_out += THREADS;",
    },
    "onehot0": map[string]string{
        "arguments": "const int* onehot$0_in",
        "inits": "onehot$0_in += tid;",
        "loads": 
            "int onehot$0 = __ldg(onehot$0_in);\n" +
            "onehot$0_in += THREADS;",
    },
    "onehot1": map[string]string{
        "arguments": "const int* onehot$0_in",
        "inits": "int onehot$0 = __ldg(onehot$0_in + bid);\n",
        "loads": "",
    },
    "const": map[string]string{
        "arguments": "float c$0",
    },
}

var isFinite = `
float $0;
asm("{\n\t"
    ".reg .pred is_finite;\n\t"
    "testp.finite.f32 is_finite, %1;\n\t"
    "selp.f32 %0, 0F3f800000, 0F00000000, is_finite;\n\t"
    "}" : "=f"($0) : "f"($1));
`

type floatOp struct {
    numOps int
    opCode string
}

// Note: binary operands come off the stack in reverse order
var floatOps = map[backends.Op]floatOp{
    backends.Assign: floatOp{2, "unused"},
    backends.Add: floatOp{2, "float $0 = $2 + $1;"},
    backends.Sub: floatOp{2, "float $0 = $2 - $1;"},
    backends.Mul: floatOp{2, "float $0 = $2 * $1;"},
    backends.Div: floatOp{2, "float $0 = $2 / $1;"},
    backends.Eq: floatOp{2, "float $0 = $2 == $1;"},
    backends.Ne: floatOp{2, "float $0 = $2 != $1;"},
    backends.Lt: floatOp{2, "float $0 = $2 < $1;"},
    backends.Le: floatOp{2, "float $0 = $2 <= $1;"},
    backends.Gt: floatOp{2, "float $0 = $2 > $1;"},
    backends.Ge: floatOp{2, "float $0 = $2 >= $1;"},
    backends.Minimum: floatOp{2, "float $0 = fminf($2, $1);"},
    backends.Maximum: floatOp{2, "float $0 = fmaxf($2, $1);"},
    backends.Pow: floatOp{2, "float $0 = powf($2, $1);"},
    backends.Finite: floatOp{1, isFinite},
    backends.Neg: floatOp{1, "float $0 = -$1;"},
    backends.Abs: floatOp{1, "float $0 = abs($1);"},
    backends.Sgn: floatOp{1, "float $0 = ($1 == 0.0f) ? 0.0f : copysignf(1.0f, $1);"},
    backends.Sqrt: floatOp{1, "float $0 = sqrtf($1);"},
    backends.Sqr: floatOp{1, "float $0 = $1 * $1;"},
    backends.Exp: floatOp{1, "float $0 = expf($1);"},
    backends.Log: floatOp{1, "float $0 = logf($1);"},
    backends.Safelog: floatOp{1, "float $0 = ($1 > 0.0f) ? logf($1) : -50.0f;"},
    backends.Exp2: floatOp{1, "float $0 = exp2f($1);"},
    backends.Log2: floatOp{1, "float $0 = log2f($1);"},
    backends.Sig: floatOp{1, "float $0 = 1.0f / (1.0f + expf(-$1));"},
    backends.Sig2: floatOp{1, "float $0 = 1.0f / (1.0f + exp2f(-$1));"},
    backends.Tanh: floatOp{1, "float $0 = tanhf($1);"},
    backends.Tanh2: floatOp{1, "float $0 = (exp2f(2.0f * $1) - 1.0f) / (exp2f(2.0f * $1) + 1.0f);"},
    backends.Rand: floatOp{0, "float $0 = frand(lfsr0, lfsr1, lfsr2);"},
    backends.Onehot: floatOp{0, "float $0 = $1 == $2;"},
} 

func getFloatOp(op backends.Op) *floatOp {
    templ, ok := floatOps[op]
    if !ok {
        return nil
    }
    return &templ
}

type reductionOp map[string]string

var reductionOps = map[backends.Op]reductionOp{
    backends.Sum: map[string]string{
        "inits": "float $0 = 0.0f;",
        "ops": "$0 += $1;",
        "shfl_red": "$0 += __shfl_xor_sync(0xffffffff, $0, i);",
        "share1_red": "sPartials[tid] += sPartials[tid + a];",
        "share2_red": "$0 = sPartials[tid] + sPartials[tid + 32];",
    },
    backends.Max: map[string]string{
        "inits": "float $0 = -FLT_MAX;",
        "ops": "$0 = fmaxf($0, $1);",
        "shfl_red": "$0 = fmaxf($0, __shfl_xor_sync(0xffffffff, $0, i));",
        "share1_red": "sPartials[tid] = fmaxf(sPartials[tid], sPartials[tid + a]);",
        "share2_red": "$0 = fmaxf(sPartials[tid], sPartials[tid + 32]);",
    },
    backends.Min: map[string]string{
        "inits": "float $0 = FLT_MAX;",
        "ops": "$0 = fminf($0, $1);",
        "shfl_red": "$0 = fminf($0, __shfl_xor_sync(0xffffffff, $0, i));",
        "share1_red": "sPartials[tid] = fminf(sPartials[tid], sPartials[tid + a]);",
        "share2_red": "$0 = fminf(sPartials[tid], sPartials[tid + 32]);",
    },
    backends.Argmax: map[string]string{
        "inits": 
            "int $0 = -1;\n" + 
            "float max = -FLT_MAX;",
        "ops": 
            "if ($1 > max) {\n" +
                "max = $1;\n" +
                "$0 = i;\n" + 
            "}",
        "shfl_red": 
            "float max2 = __shfl_xor_sync(0xffffffff, max, i);\n" +
            "int argMax2 = __shfl_xor_sync(0xffffffff, $0, i);\n" +
            "if (max2 > max) {\n" +
                "max = max2;\n" +
                "$0 = argMax2;\n" +
            "} else if (max2 == max && argMax2 < $0) {\n" +
                "$0 = argMax2;\n" +
            "}",
    },
    backends.Argmin: map[string]string{
        "inits": 
            "int $0 = -1;\n" + 
            "float min = FLT_MAX;",
        "ops": 
            "if ($1 < min) {\n" +
                "min = $1;\n" +
                "$0 = i;\n" +
            "}",
        "shfl_red": 
            "float min2 = __shfl_xor_sync(0xffffffff, min, i);\n" + 
            "int argMin2 = __shfl_xor_sync(0xffffffff, $0, i);\n" +
            "if (min2 < min) {\n" +
                "min = min2;\n" +
                "$0 = argMin2;\n" + 
            "} else if (min2 == min && argMin2 < $0) {\n" +
                "$0 = argMin2;\n" +
            "}",
    },
} 

func getReductionOp(op backends.Op) reductionOp {
    templ, ok := reductionOps[op]
    if !ok {
        return nil
    }
    return templ
}

