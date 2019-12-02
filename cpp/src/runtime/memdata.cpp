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

namespace arhat {

//
//    MemoryData
//

// construction/destruction

MemoryData::MemoryData() {
    len = 0;
    cap = 0;
    data = nullptr;
}

MemoryData::~MemoryData() {
    Destroy();
}

// interface

void MemoryData::Reset() {
    Destroy();
    len = 0;
    cap = 0;
    data = nullptr;
}

int MemoryData::Len() const {
    return len;
}

int MemoryData::Size(int index) const {
    if (!(index >= 0 && index < len)) {
        IndexError(__FILE__, __LINE__);
    }
    return data[index].size;
}

byte_t *MemoryData::Buffer(int index) const {
    if (!(index >= 0 && index < len)) {
        IndexError(__FILE__, __LINE__);
    }
    return data[index].buf;
}

void MemoryData::GetData(int index, void *buf) const {
    if (!(index >= 0 && index < len)) {
        IndexError(__FILE__, __LINE__);
    }
    memcpy(buf, data[index].buf, data[index].size);
}

void MemoryData::Add(int size, const void *buf) {
    byte_t *p = new byte_t[size];
    memcpy(p, buf, size);
    AddEntry(size, p);
}

void MemoryData::Load(const char *fname) {
    FILE *fp = fopen(fname, "rb");
    if (fp == nullptr) {
        OpenError(fname, __FILE__, __LINE__);
    }
    int n = ReadInt(fp);
    for (int i = 0; i < n; i++) {
        int size = ReadInt(fp);
        byte_t *p = new byte_t[size];
        ReadBuffer(fp, size, p);
        AddEntry(size, p);
    }
    fclose(fp);
}

void MemoryData::Save(const char *fname) {
    FILE *fp = fopen(fname, "wb");
    if (fp == nullptr) {
        OpenError(fname, __FILE__, __LINE__);
    }
    WriteInt(fp, len);
    for (int i = 0; i < len; i++) {
        int size = data[i].size;
        WriteInt(fp, size);
        WriteBuffer(fp, size, data[i].buf);
    }
    fclose(fp);
}

// implementation

void MemoryData::AddEntry(int size, byte_t *buf) {
    if (len >= cap) {
        int newCap = (cap > 0) ? 2 * cap : 1024;
        Entry *newData = new Entry[newCap];
        for (int i = 0; i < len; i++) {
            newData[i] = data[i];
        }
        delete[] data;
        cap = newCap;
        data = newData;
    }
    data[len].size = size;
    data[len].buf = buf;
    len++;
}

void MemoryData::Destroy() {
    for (int i = len - 1; i >= 0; i--) {
        delete[] data[i].buf;
    }
    delete[] data;
}

// simplified: raw I/O, no platform-agnostic serialization

int MemoryData::ReadInt(FILE *fp) {
    int64_t n;
    if (fread(&n, sizeof(n), 1, fp) != 1) {
        fclose(fp);
        EofError(__FILE__, __LINE__);
    }
    return int(n);
}

void MemoryData::WriteInt(FILE *fp, int val) {
    int64_t n = int64_t(val);
    fwrite(&n, sizeof(n), 1, fp);
}

void MemoryData::ReadBuffer(FILE *fp, int size, void *buf) {
    if (fread(buf, 1, size, fp) != size) {
        fclose(fp);
        EofError(__FILE__, __LINE__);
    }
}

void MemoryData::WriteBuffer(FILE *fp, int size, const void *buf) {
    fwrite(buf, 1, size, fp);
}

void MemoryData::OpenError(const char *fname, const char *file, int line) {
    char buf[256];
    sprintf(buf, "Cannot open file [%s]", fname);
    Error(buf, file, line);
}

void MemoryData::EofError(const char *file, int line) {
    Error("Unexpected end of file", file, line);
}

void MemoryData::IndexError(const char *file, int line) {
    Error("Index out of range", file, line);
}

} // arhat

