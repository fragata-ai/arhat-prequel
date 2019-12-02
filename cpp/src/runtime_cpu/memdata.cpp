//
// Copyright 2019 FRAGATA COMPUTER SYSTEMS AG
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
#include <cstdint>
#include <cstring>
#include <cassert>

#include "runtime/arhat.h"
#include "runtime_cpu/arhat.h"

namespace arhat {
namespace cpu {

//
//    MemoryReader
//

// construction/destruction

MemoryReader::MemoryReader() {
    pos = 0;
}

MemoryReader::~MemoryReader() { }

// interface

void MemoryReader::Start() {
    pos = 0;
}

void MemoryReader::Read(int size, void *buf) {
    if (Size(pos) != size) {
        Error("Buffer size mismatch", __FILE__, __LINE__);
    }
    void *p = Buffer(pos);
    CpuMemcpy(buf, p, size);
}

//
//    MemoryWriter
//

// construction/destruction

MemoryWriter::MemoryWriter() { }

MemoryWriter::~MemoryWriter() { }

// interface

void MemoryWriter::Write(int size, const void *buf) {
    byte_t *p = new byte_t[size];
    CpuMemcpy(p, buf, size);
    AddEntry(size, p);
}

} // cpu
} // arhat

