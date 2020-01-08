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
    "fragata/arhat/base"
    "strings"
)

// exported names (used by CUDA backends)

var (
    // data
    InitRandFunc = initRandFunc
    InitRandRoundFunc = initRandRoundFunc
    FinishRandFunc = finishRandFunc
    CommonKepler = commonKepler
    CommonUrandGen = commonUrandGen
    CommonFrand = commonFrand
    CommonRound = commonRound
    CommonFp16toFp32 = commonFp16toFp32
    CommonMaxAbs = commonMaxAbs
    EwTypes = ewTypes
    EwStringsRound = ewStringsRound
    // functions
    Format = format
)

var initRandFunc = `
    unsigned lfsr0, lfsr1, lfsr2;
    unsigned idx = bid * THREADS + tid;
    rand_state += idx % RAND_POOL_SIZE;
    lfsr0 = *(rand_state + 0*RAND_POOL_SIZE);
    lfsr1 = *(rand_state + 1*RAND_POOL_SIZE);
    lfsr2 = *(rand_state + 2*RAND_POOL_SIZE);
`

var initRandRoundFunc = `
    int i_rand_scale = (127 - 32 - mantissa_bits) << 23;
    float rand_scale = *(float*)&i_rand_scale;
    unsigned rand_mask = 0xffffffff << (23 - mantissa_bits);
`

var finishRandFunc = `
    *(rand_state + 0*RAND_POOL_SIZE) = lfsr0;
    *(rand_state + 1*RAND_POOL_SIZE) = lfsr1;
    *(rand_state + 2*RAND_POOL_SIZE) = lfsr2;
`

var commonKepler = `
#define __ldg(x) (*(x))
`

var commonUrandGen = `
__device__ unsigned urand_gen(unsigned& lfsr0, unsigned& lfsr1, unsigned& lfsr2)
{
    lfsr0 = ((lfsr0 & 0xfffffffe) << 12) ^ (((lfsr0 << 13) ^ lfsr0) >> 19);
    lfsr1 = ((lfsr1 & 0xfffffff8) <<  4) ^ (((lfsr1 << 2)  ^ lfsr1) >> 25);
    lfsr2 = ((lfsr2 & 0xfffffff0) << 11) ^ (((lfsr2 << 3)  ^ lfsr2) >> 11);
    return lfsr0 ^ lfsr1 ^ lfsr2;
}
`

var commonFrand = `
__device__ __forceinline__ float frand(unsigned& lfsr0, unsigned& lfsr1, unsigned& lfsr2)
{
    unsigned urand = urand_gen(lfsr0, lfsr1, lfsr2);
    float val;
    asm("cvt.rn.f32.u32 %0, %1;\n\t"
        "mul.f32 %0, %0, 0F2f800000;"
        : "=f"(val) : "r"(urand));
    return val;
}
`

var commonRound = map[string]map[base.Dtype]string{

    "random": map[base.Dtype]string{

        base.Float32: `
__device__ float fp32_to_fp32_rand(
    float val, unsigned& lfsr0, unsigned& lfsr1, unsigned& lfsr2, float rand_scale,
          unsigned rand_mask)
{
    unsigned urand = urand_gen(lfsr0, lfsr1, lfsr2);

    float ret;
    asm("{\n\t"
        ".reg .f32 exponent, frand, result;\n\t"
        "and.b32 exponent, %1, 0xff800000;\n\t"
        "mul.f32 exponent, exponent, %2;\n\t"
        "cvt.rz.f32.u32 frand, %3;\n\t"
        "fma.rz.f32 result, exponent, frand, %1;\n\t"
        "and.b32 %0, result, %4;\n\t"
        "}" : "=f"(ret) : "f"(val), "f"(rand_scale), "r"(urand), "r"(rand_mask));

    return ret;
}
`,
        base.Float16: `
__device__ unsigned short fp32_to_fp16_rand(
    float val, unsigned& lfsr0, unsigned& lfsr1, unsigned& lfsr2, float rand_scale,
          unsigned rand_mask)
{
    unsigned urand = urand_gen(lfsr0, lfsr1, lfsr2);

    unsigned short half;
    asm("{\n\t"
        ".reg .f16 result16;\n\t"
        ".reg .f32 exponent, frand, result32;\n\t"
        "and.b32 exponent, %1, 0xff800000;\n\t"
        "mul.f32 exponent, exponent, %2;\n\t"
        "cvt.rz.f32.u32 frand, %3;\n\t"
        "fma.rz.f32 result32, exponent, frand, %1;\n\t"
        "and.b32 result32, result32, %4;\n\t"
        "cvt.rz.f16.f32 result16, result32;\n\t"
        "mov.b16 %0, result16;\n\t"
        "}" : "=h"(half) : "f"(val), "f"(rand_scale), "r"(urand), "r"(rand_mask));

    return half;
}
`,
        base.Int32: `
__device__ __forceinline__ int fp32_to_int32_rand(
    float val, unsigned& lfsr0, unsigned& lfsr1, unsigned& lfsr2)
{
    unsigned urand = urand_gen(lfsr0, lfsr1, lfsr2);
    int ret;
    asm("{\n\t"
        ".reg .f32 frand, result32;\n\t"
        "cvt.rz.f32.u32 frand, %2;\n\t"
        "copysign.f32 frand, %1, frand;\n\t"
        "mul.rz.f32 frand, frand, 0F2f800000;\n\t"
        "add.rz.f32 result32, frand, %1;\n\t"
        "cvt.rzi.s32.f32 %0, result32;\n\t"
        "}" : "=r"(ret) : "f"(val), "r"(urand));
    return ret;
}
`,
        base.Int16: `
__device__ __forceinline__ short fp32_to_int16_rand(
    float val, unsigned& lfsr0, unsigned& lfsr1, unsigned& lfsr2)
{
    unsigned urand = urand_gen(lfsr0, lfsr1, lfsr2);
    short half;
    asm("{\n\t"
        ".reg .f32 frand, result32;\n\t"
        "cvt.rz.f32.u32 frand, %2;\n\t"
        "copysign.f32 frand, %1, frand;\n\t"
        "mul.rz.f32 frand, frand, 0F2f800000;\n\t"
        "add.rz.f32 result32, frand, %1;\n\t"
        "cvt.rzi.s16.f32 %0, result32;\n\t"
        "}" : "=h"(half) : "f"(val), "r"(urand));
    return half;
}
`,
        base.Int8: `
__device__ __forceinline__ char fp32_to_int8_rand(
    float val, unsigned& lfsr0, unsigned& lfsr1, unsigned& lfsr2)
{
    unsigned urand = urand_gen(lfsr0, lfsr1, lfsr2);
    int ret;
    asm("{\n\t"
        ".reg .f32 frand, result32;\n\t"
        "cvt.rz.f32.u32 frand, %2;\n\t"
        "copysign.f32 frand, %1, frand;\n\t"
        "mul.rz.f32 frand, frand, 0F2f800000;\n\t"
        "add.rz.f32 result32, frand, %1;\n\t"
        "cvt.rzi.s8.f32 %0, result32;\n\t"
        "}" : "=r"(ret) : "f"(val), "r"(urand));
    return ret;
}
`,
    },
    "nearest": map[base.Dtype]string{

        base.Float16: `
__device__ __forceinline__ unsigned short fp32_to_fp16(float val)
{
    unsigned short ret;
    asm("{\n\t"
        ".reg .f16 f16;\n\t"
        "cvt.rn.f16.f32 f16, %1;"
        "mov.b16 %0, f16;\n\t"
        "}" : "=h"(ret) : "f"(val));
    return ret;
}
`,
        base.Int32: `
__device__ __forceinline__ int fp32_to_int32(float val)
{
    int ret;
    asm("cvt.rni.s32.f32 %0, %1;" : "=r"(ret) : "f"(val));
    return ret;
}
`,
        base.Uint32: `
__device__ __forceinline__ unsigned fp32_to_uint32(float val)
{
    unsigned ret;
    asm("cvt.rni.u32.f32 %0, %1;" : "=r"(ret) : "f"(val));
    return ret;
}
`,
        base.Int16: `
__device__ __forceinline__ short fp32_to_int16(float val)
{
    short ret;
    asm("cvt.rni.s16.f32 %0, %1;" : "=h"(ret) : "f"(val));
    return ret;
}
`,
        base.Uint16: `
__device__ __forceinline__ unsigned short fp32_to_uint16(float val)
{
    unsigned short ret;
    asm("cvt.rni.u16.f32 %0, %1;" : "=h"(ret) : "f"(val));
    return ret;
}
`,
        base.Int8: `
__device__ __forceinline__ char fp32_to_int8(float val)
{
    int ret;
    asm("cvt.rni.s8.f32 %0, %1;" : "=r"(ret) : "f"(val));
    return ret;
}
`,
        base.Uint8: `
__device__ __forceinline__ unsigned char fp32_to_uint8(float val)
{
    unsigned ret;
    asm("cvt.rni.u8.f32 %0, %1;" : "=r"(ret) : "f"(val));
    return ret;
}
`,
    },
}

func commonRoundInit() {
    // random rounding not yet used for these types
    commonRound["random"][base.Uint32] = commonRound["nearest"][base.Uint32]
    commonRound["random"][base.Uint16] = commonRound["nearest"][base.Uint16]
    commonRound["random"][base.Uint8] = commonRound["nearest"][base.Uint8]

/* TODO: Revise this
    commonRound["random"][base.Fixed32] = commonRound["random"][base.Int32] 
    commonRound["random"][base.Fixed16] = commonRound["random"][base.Int16] 
    commonRound["random"][base.Fixed8] = commonRound["random"][base.Int8] 

    commonRound["nearest"][base.Fixed32] = commonRound["nearest"][base.Int32] 
    commonRound["nearest"][base.Fixed16] = commonRound["nearest"][base.Int16] 
    commonRound["nearest"][base.Fixed8] = commonRound["nearest"][base.Int8] 
*/
    }

var commonFp16toFp32 = `
__device__ __forceinline__ float fp16_to_fp32(unsigned short val)
{
    float ret;
    asm("{\n\t"
        ".reg .f16 f16;\n\t"
        "mov.b16 f16, %1;\n\t"
        "cvt.f32.f16 %0, f16\n\t;"
        "}" : "=f"(ret) : "h"(val));
    return ret;
}
`

var commonMaxAbs = `
__device__ __forceinline__ int max_abs(int max_abs, int val)
{
    asm("{\n\t"
        ".reg .s32 abs_val;\n\t"
        "abs.s32 abs_val, %1;\n\t"
        "max.s32 %0, %0, abs_val;\n\t"
        "}" : "+r"(max_abs) : "r"(val));
    return max_abs;
}
`

var ewTypes = map[base.Dtype]map[string]string{
    base.Float32: map[string]string{
        "type": "float",
        "type4": "float4",
        "cvt": "",
        "cvt_out": "",
    },
    base.Float16: map[string]string{
        "type": "unsigned short",
        "type4": "ushort4",
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
/* TODO: Revise this
    base.Fixed32: map[string]string{
        "type": "int",
        "cvt": "scale$0 * (float)",
    },
    base.Fixed16: map[string]string{
        "type": "short",
        "cvt": "scale$0 * (float)",
    },
    base.Fixed8: {
        "type": "char",
        "cvt": "scale$0 * (float)",
    },
*/
}

var ewStringsRound = map[string]map[base.Dtype]string{
    "random": map[base.Dtype]string{
        base.Float32: 
            "float $0 = fp32_to_fp32_rand($1, lfsr0, lfsr1, " +
                "lfsr2, rand_scale, rand_mask);",
        base.Float16: 
            "unsigned short $0 = fp32_to_fp16_rand($1, lfsr0, lfsr1, " +
                "lfsr2, rand_scale, rand_mask);",
        base.Uint32: "unsigned int $0 = fp32_to_uint32($1);",
        base.Uint16: "unsigned short $0 = fp32_to_uint16($1);",
        base.Uint8: "unsigned char $0 = fp32_to_uint8($1);",
        base.Int32: "int $0 = fp32_to_int32_rand($1, lfsr0, lfsr1, lfsr2);",
        base.Int16: "short $0 = fp32_to_int16_rand($1, lfsr0, lfsr1, lfsr2);",
        base.Int8: "char $0 = fp32_to_int8_rand($1, lfsr0, lfsr1, lfsr2);",
/* TODO: Revise this
        base.Fixed32: "int $0 = fp32_to_int32_rand($1, lfsr0, lfsr1, lfsr2);",
        base.Fixed16: "short $0 = fp32_to_int16_rand($1, lfsr0, lfsr1, lfsr2);",
        base.Fixed8: "char $0 = fp32_to_int8_rand($1, lfsr0, lfsr1, lfsr2);",
*/
    },
    "nearest": map[base.Dtype]string{
        base.Float16: "unsigned short $0 = fp32_to_fp16($1);",
        base.Uint32: "unsigned int $0 = fp32_to_uint32($1);",
        base.Uint16: "unsigned short $0 = fp32_to_uint16($1);",
        base.Uint8: "unsigned char $0 = fp32_to_uint8($1);",
        base.Int32: "int $0 = fp32_to_int32($1);",
        base.Int16: "short $0 = fp32_to_int16($1);",
        base.Int8: "char $0 = fp32_to_int8($1);",
/* TODO: Revise this
        base.Fixed32: "int $0 = fp32_to_int32($1);",
        base.Fixed16: "short $0 = fp32_to_int16($1);",
        base.Fixed8: "char $0 = fp32_to_int8($1);",
*/
    },
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

func init() {
    commonRoundInit()
}

