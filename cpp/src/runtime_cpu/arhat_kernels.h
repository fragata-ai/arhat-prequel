#pragma once

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
// Ported from Python to C++ and partly modified by FRAGATA COMPUTER SYSTEMS AG.
//

#include <cstdint>
#include <cfloat>
#include <cmath>

//
//    Random number generation and rounding
//

// commonFrand

float frand();

// commonRound

unsigned short fp32_to_fp16(float val);

inline int fp32_to_int32(float val) {
    return (int)(int32_t)std::round(val);
}

inline unsigned fp32_to_uint32(float val) {
    return (unsigned)(uint32_t)std::round(val);
}

inline short fp32_to_int16(float val) {
    return (short)(int16_t)std::round(val);
}

inline unsigned short fp32_to_uint16(float val) {
    return (unsigned short)(uint16_t)std::round(val);
}

inline char fp32_to_int8(float val) {
    return (char)(int8_t)std::round(val);
}

inline unsigned char fp32_to_uint8(float val) {
    return (unsigned char)(int8_t)std::round(val);
}

inline float fp32_to_fp32_rand(
        float val, 
        float rand_scale, 
        unsigned rand_mask) {
    // not yet implemented
    return val;
}

inline unsigned short fp32_to_fp16_rand(
        float val, 
        float rand_scale,
        unsigned rand_mask) {
    // not yet implemented
    return fp32_to_fp16(val);
}

inline int fp32_to_int32_rand(float val) {
    // not yet implemented
    return fp32_to_int32(val);
}

inline short fp32_to_int16_rand(float val) {
    // not yet implemented
    return fp32_to_int16(val);
}

inline char fp32_to_int8_rand(float val) {
    // not yet implemented
    return fp32_to_int8(val);
}

// commonFp16toFp32

float fp16_to_fp32(unsigned short val);

// isFinite

inline float isFinite(float x) {
    return std::isfinite(x);
}

