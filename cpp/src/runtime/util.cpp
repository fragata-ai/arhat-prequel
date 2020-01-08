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
// Ported from Python to C++ and partly modified by FRAGATA COMPUTER SYSTEMS AG.
//

#include <cstdint>
#include <cstring>
#include <cassert>

#include "runtime/arhat.h"

namespace arhat {

//
//    Array utilities
//

namespace {

void TransposeSlice8(uint8_t *dst, const uint8_t *src, int dim0, int dim1) {
    for (int i = 0; i < dim0; i++) {
        for (int k = 0; k < dim1; k++) {
            dst[k * dim0 + i] = src[i * dim1 + k];
        }
    }
}

void TransposeSlice16(uint16_t *dst, const uint16_t *src, int dim0, int dim1) {
    for (int i = 0; i < dim0; i++) {
        for (int k = 0; k < dim1; k++) {
            dst[k * dim0 + i] = src[i * dim1 + k];
        }
    }
}

void TransposeSlice32(uint32_t *dst, const uint32_t *src, int dim0, int dim1) {
    for (int i = 0; i < dim0; i++) {
        for (int k = 0; k < dim1; k++) {
            dst[k * dim0 + i] = src[i * dim1 + k];
        }
    }
}

void TransposeSlice64(uint64_t *dst, const uint64_t *src, int dim0, int dim1) {
    for (int i = 0; i < dim0; i++) {
        for (int k = 0; k < dim1; k++) {
            dst[k * dim0 + i] = src[i * dim1 + k];
        }
    }
}

} // namespace

void TransposeSlice(void *dst, const void *src, int dim0, int dim1, int itemSize) {
    switch (itemSize) {
    case 1:
        TransposeSlice8((uint8_t *)dst, (const uint8_t *)src, dim0, dim1);
        break;
    case 2:
        TransposeSlice16((uint16_t *)dst, (const uint16_t *)src, dim0, dim1);
        break;
    case 4:
        TransposeSlice32((uint32_t *)dst, (const uint32_t *)src, dim0, dim1);
        break;
    case 8:
        TransposeSlice64((uint64_t *)dst, (const uint64_t *)src, dim0, dim1);
        break;
    default:
        assert(false);
    }
}

} // arhat

