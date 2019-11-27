
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
    xfer0 = new char[xferSize];
    xfer1 = new char[xferSize];
}

int ArrayWriter::Len() {
    return data.Len();
}

void *ArrayWriter::Buffer(int index) {
    return data.Buffer(index);
}

// overrides

void ArrayWriter::WriteBatch(const void *buf, int count) {
    assert(count <= bsz);
    // full batch must be always available even when (count < bsz)
    CpuMemcpy(xfer0, buf, dim1 * bsz * itemSize);
    TransposeSlice(xfer1, xfer0, dim1, bsz, itemSize);
    char *ptr = (char *)xfer1;
    int rsz = dim1 * itemSize;
    for (int i = 0; i < count; i++) {
        data.Add(rsz, ptr);
        ptr += rsz;
    }
}

} // cpu
} // arhat

