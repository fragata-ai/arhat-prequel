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

#include <cstdio>
#include <cassert>

#include "runtime/arhat.h"
#include "runtime_cpu/arhat.h"

namespace arhat {
namespace cpu {

//
//    ArrayWriter
//

// construction/destruction

ArrayWriter::ArrayWriter() {
    dim1 = 0;
    bsz = 0;
    itemSize = 0;
    xfer0 = nullptr;
    xfer1 = nullptr;
}

ArrayWriter::~ArrayWriter() {
    delete[] xfer0;
    delete[] xfer1;
}

// interface

void ArrayWriter::Init(int dim1, int bsz, int itemSize) {
    assert(xfer0 == nullptr && xfer1 == nullptr);
    assert(itemSize == 1 || itemSize == 2 || itemSize == 4 || itemSize == 8);
    this->dim1 = dim1;
    this->bsz = bsz;
    this->itemSize = itemSize;
    int xferSize = dim1 * bsz * itemSize;
    xfer0 = new byte_t[xferSize];
    xfer1 = new byte_t[xferSize];
}

int ArrayWriter::Len() const {
    return data.Len();
}

int ArrayWriter::Size(int index) const {
    return data.Size(index);
}

byte_t *ArrayWriter::Buffer(int index) const {
    return data.Buffer(index);
}

void ArrayWriter::GetData(int index, void *buf) const {
    data.GetData(index, buf);
}

// overrides

void ArrayWriter::WriteBatch(const void *buf, int count) {
    assert(count <= bsz);
    // full batch must be always available even when (count < bsz)
    CpuMemcpy(xfer0, buf, dim1 * bsz * itemSize);
    TransposeSlice(xfer1, xfer0, dim1, bsz, itemSize);
    byte_t *ptr = xfer1;
    int rsz = dim1 * itemSize;
    for (int i = 0; i < count; i++) {
        data.Add(rsz, ptr);
        ptr += rsz;
    }
}

} // cpu
} // arhat

