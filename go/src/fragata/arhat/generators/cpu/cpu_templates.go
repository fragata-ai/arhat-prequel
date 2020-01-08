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

package cpu

import (
    "fmt"
    "fragata/arhat/backends"
    "fragata/arhat/base"
    "strings"
)

var initRandRoundFunc = `
    int i_rand_scale = (127 - 32 - mantissa_bits) << 23;
    float rand_scale = *(float*)&i_rand_scale;
    unsigned rand_mask = 0xffffffff << (23 - mantissa_bits);
`

var ewTypes = map[base.Dtype]map[string]string{
    base.Float32: map[string]string{
        "type": "float",
        "cvt": "",
        "cvt_out": "",
    },
    base.Float16: map[string]string{
        "type": "unsigned short",
        "cvt": "fp16_to_fp32",
        "cvt_out": "fp32_to_fp16",
    },
    base.Int32: map[string]string{
        "type": "int",
        "cvt": "(float)",
    },
    base.Uint32: map[string]string{
        "type": "unsigned int",
        "cvt": "(float)",
    },
    base.Int16: map[string]string{
        "type": "short",
        "cvt": "(float)",
    },
    base.Uint16: map[string]string{
        "type": "unsigned short",
        "cvt": "(float)",
    },
    base.Int8: map[string]string{
        "type": "char",
        "cvt": "(float)",
    },
    base.Uint8: map[string]string{
        "type": "unsigned char",
        "cvt": "(float)",
    },
}

var ewStringsRound = map[string]map[base.Dtype]string{
    "random": map[base.Dtype]string{
        base.Float32: "float $0 = fp32_to_fp32_rand($1, rand_scale, rand_mask);",
        base.Float16: "unsigned short $0 = fp32_to_fp16_rand($1, rand_scale, rand_mask);",
        base.Uint32: "unsigned int $0 = fp32_to_uint32($1);",
        base.Uint16: "unsigned short $0 = fp32_to_uint16($1);",
        base.Uint8: "unsigned char $0 = fp32_to_uint8($1);",
        base.Int32: "int $0 = fp32_to_int32_rand($1);",
        base.Int16: "short $0 = fp32_to_int16_rand($1);",
        base.Int8: "char $0 = fp32_to_int8_rand($1);",
    },
    "nearest": map[base.Dtype]string{
        base.Float16: "unsigned short $0 = fp32_to_fp16($1);",
        base.Uint32: "unsigned int $0 = fp32_to_uint32($1);",
        base.Uint16: "unsigned short $0 = fp32_to_uint16($1);",
        base.Uint8: "unsigned char $0 = fp32_to_uint8($1);",
        base.Int32: "int $0 = fp32_to_int32($1);",
        base.Int16: "short $0 = fp32_to_int16($1);",
        base.Int8: "char $0 = fp32_to_int8($1);",
    },
}

var ewTemplate = `
void {{.name}}(
    int blocks,
    {{.arguments}})
{
    for (int bid = 0; bid < blocks; bid++) {
        {{.inits}}
` 

var stageTemplate = map[string]string{
    "loop": `

     for (int i = 0; i < n$0; i++) {
        {{.loads$0}}

        {{.ops$0}}
    }
`,

    "red_ops": `

        {{.ops$0}}
`,

    "red_out": `

        {{.ops$0}}
`,
} 

var finTemplate = `
        {{.finish}}
    }
}
`

var ewStrings = map[string]map[string]string{
    // 0: argId, 1: stage, 2: type, 3: cvt
    "in0": map[string]string{
        "arguments": "const $2* a$0_in, int row_strd$0, int col_strd$0",
        "inits": "const $2* a$0_in$1 = a$0_in + bid * row_strd$0;\n",
        "loads": 
            "float a$0 = $3(*a$0_in$1);\n" +
            "a$0_in$1 += col_strd$0;",
    },
    "in1": map[string]string{
        "arguments": "const $2* a$0_in, int row_strd$0, int col_strd$0, const int* take$0_in",
        "inits": "const $2* a$0_in$1 = a$0_in + (*take$0_in + bid) * row_strd$0;\n",
        "loads": 
            "float a$0 = $3(*a$0_in$1);\n" +
            "a$0_in$1 += col_strd$0;",
    },
    "in2": map[string]string{
        "arguments": "const $2* a$0_in, int row_strd$0, int col_strd$0, const int* take$0_in",
        "inits": 
            "const $2* a$0_in$1 = a$0_in + bid * row_strd$0;\n" +
            "const int* take$0_in$1 = take$0_in;",
        "loads": 
            "float a$0 = $3(*(a$0_in$1 + (*take$0_in$1)));\n" +
            "take$0_in$1++;",
    },
    "out0": map[string]string{
        "arguments": "$2* a_out, int row_strd, int col_strd",
        "inits": "$2* a_out0 = a_out + bid * row_strd;",
        "output": 
            "*a_out0 = $0;\n" +
            "a_out0 += col_strd;",
    },
    "out1": map[string]string{
        "arguments": "$2* a_out, int row_strd, int col_strd, const int* take_out",
        "inits": "$2* a_out1 = a_out + (*take_out + bid) * row_strd;\n",
        "output": 
            "*a_out1 = $0;\n" +
            "a_out1 += col_strd;",
    },
    "out2": map[string]string{
        "arguments": "$2* a_out, int row_strd, int col_strd, const int* take_out",
        "inits": "$2* a_out2 = a_out + bid * row_strd;",
        "output": 
            "*(a_out2 + (*take_out)) = $0;\n" +
            "take_out++;",
    },
    "onehot0": map[string]string{
        "arguments": "const int* onehot$0_in",
        "inits": "",
        "loads": 
            "int onehot$0 = (*onehot$0_in);\n" +
            "onehot$0_in++;",
    },
    "onehot1": map[string]string{
        "arguments": "const int* onehot$0_in",
        "inits": "int onehot$0 = (*onehot$0_in + bid);\n",
        "loads": "",
    },
    "const": map[string]string{
        "arguments": "float c$0",
    },
}

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
    backends.Finite: floatOp{1, "float $0 = isFinite($1);"},
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
    backends.Rand: floatOp{0, "float $0 = frand();"},
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
    },
    backends.Max: map[string]string{
        "inits": "float $0 = -FLT_MAX;",
        "ops": "$0 = fmaxf($0, $1);",
    },
    backends.Min: map[string]string{
        "inits": "float $0 = FLT_MAX;",
        "ops": "$0 = fminf($0, $1);",
    },
    backends.Argmax: map[string]string{
        "inits": "int $0 = -1; float max = -FLT_MAX;",
        "ops": 
            "if ($1 > max) {\n" +
                "max = $1;\n" +
                "$0 = i;\n" + 
            "}",
    },
    backends.Argmin: map[string]string{
        "inits": "int $0 = -1; float min = FLT_MAX;",
        "ops": 
            "if ($1 < min) {\n" +
                "min = $1;\n" +
                "$0 = i;\n" +
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

func format(s string, args ...interface{}) string {
    var b strings.Builder
    n := len(s)
    for  i := 0; i < n; i++ {
        if c := s[i]; c != '$' {
            b.WriteByte(c)
        } else {
            i++
            if c = s[i]; c >= '0' && c <= '9' {
                k := int(c-'0')
                b.WriteString(fmt.Sprintf("%v", args[k])) 
            } else {
                b.WriteByte('$')
                b.WriteByte(c)
            }
        }
    }
    return b.String()
}

