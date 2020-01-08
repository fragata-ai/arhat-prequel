//
// Copyright 2019-2020 FRAGATA COMPUTER SYSTEMS AG
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

#include <cstring>
#include <cassert>

#include "runtime/arhat.h"
#include "runtime_cpu/arhat.h"

namespace arhat {
namespace cpu {

//
//    Global device memory
//

void *CpuMemAlloc(int bytes) {
    return new char[bytes];
}

//
//    Initializing device memory
//

void CpuMemsetD8(void *dest, uint8_t data, int count) {
    memset(dest, data, count);
}

void CpuMemsetD16(void *dest, uint16_t data, int count) {
    if (data == 0) {
        memset(dest, 0, count * sizeof(uint16_t));
    } else {
        uint16_t *p = static_cast<uint16_t *>(dest);
        for (int i = 0; i < count; i++) {
            p[i] = data;
        }
    }
}

void CpuMemsetD32(void *dest, uint32_t data, int count) {
    if (data == 0) {
        memset(dest, 0, count * sizeof(uint32_t));
    } else {
        uint32_t *p = static_cast<uint32_t *>(dest);
        for (int i = 0; i < count; i++) {
            p[i] = data;
        }
    }
}

//
//    Unstructured memory transfers
//

void CpuMemcpy(void *dest, const void *src, int size) {
    memcpy(dest, src, size);
}

//
//    Structured memory transfer
//

float CpuGetFloat(void *src, int nbytes) {
    assert(nbytes == 4 || nbytes == 8);
    float result = 0.0f;
    // ACHTUNG: Little endian only
    if (nbytes == 4) {
        result = *reinterpret_cast<float *>(src);
    } else if (nbytes == 8) {
        result = float(*reinterpret_cast<double *>(src));
    }
    return result;
}

//
//    Utility functions
//

float CpuGetFloatSum(float *src, int start, int stop) {
    float result = 0.0f;
    for (int i = start; i < stop; i++) {
        result += src[i];
    }
    return result;
}

void CpuGetData(void *dest, int start, int stop, void *src, int itemSize) {
    if (start >= stop) {
        return;
    }
    char *p = static_cast<char *>(dest) + start * itemSize;
    int size = (stop - start) * itemSize;
    memcpy(p, src, size);
}

} // cpu
} // arhat

