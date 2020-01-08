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

#include <cstdio>
#include <cassert>

#include "runtime/arhat.h"
#include "runtime_cuda/arhat.h"

namespace arhat {
namespace cuda {

//
//    ArrayIterator
//

// constructors/destructors

ArrayIterator::ArrayIterator() {
    ndata = 0;
    start = 0;
    nclass = 0;
    makeOnehot = false;
    xsrc = nullptr;
    xsrcDim0 = 0;
    xsrcDim1 = 0;
    ysrc = nullptr;
    ysrcDim0 = 0;
    ysrcDim1 = 0;
    bsz = 0;
    xbuf = nullptr;
    xbufDim0 = 0;
    xbufDim1 = 0;
    ybuf = nullptr;
    ybufDim0 = 0;
    ybufDim1 = 0;
    pos = 0;
}

ArrayIterator::~ArrayIterator() { 
    delete []xbuf;
    delete []ybuf;
}

// interface

void ArrayIterator::Init(
        float *x,
        int xdim0,
        int xdim1,
        float *y,
        int ydim1,
        int nclass,
        bool makeOnehot) {
    if (makeOnehot && nclass == 0 && y != nullptr)
        Error(
            "Must provide number of classes when creating onehot labels", 
                __FILE__, __LINE__);

    if (y != nullptr) {
        // for classifiction, the labels must be from 0 .. K-1, where K=nclass
        if (makeOnehot) {
            int ysize = xdim0 * ydim1;
            for (int i = 0; i < ysize; i++) {
                if (!(y[i] >= 0.0f && y[i] <= float(nclass - 1))) {
                    char buf[256];
                    sprintf(buf, "Labels must range from 0 to %d", nclass - 1);
                    Error(buf, __FILE__, __LINE__);
                }
                if (float(int(y[i])) != y[i])
                    Error("Labels must only contain integers", __FILE__, __LINE__);
            }
        }
    }

    ndata = xdim0;
    start = 0;
    this->nclass = nclass;
    this->makeOnehot = makeOnehot;

    xsrc = x;
    xsrcDim0 = xdim0;
    xsrcDim1 = xdim1;

    ysrc = y;
    if (y != nullptr) {
        ysrcDim0 = xdim0;
        ysrcDim1 = ydim1;
    } else {
        ysrcDim0 = 0;
        ysrcDim1 = 0;
    }

    bsz = 0;

    xbuf = nullptr;
    xbufDim0 = 0;
    xbufDim1 = 0;

    ybuf = nullptr;
    ybufDim0 = 0;
    ybufDim1 = 0;
}

// overrides

void ArrayIterator::SetBsz(int bsz) {
    assert(this->bsz == 0);
    this->bsz = bsz;

    xbufDim0 = xsrcDim1;
    xbufDim1 = bsz;
    xbuf = new float[xbufDim0 * xbufDim1];

    if (ysrc != nullptr) {
        if (makeOnehot)
            ybufDim0 = nclass;
        else
            ybufDim0 = ysrcDim1;
        ybufDim1 = bsz;
        ybuf = new float[ybufDim0 * ybufDim1];
    }
}

int ArrayIterator::Nbatches() {
    assert(bsz != 0);
    return (ndata + bsz - 1) / bsz;
}

int ArrayIterator::Ndata() {
    return ndata;
}

void ArrayIterator::Reset() {
    start = 0;
}

void ArrayIterator::Start() {
    pos = start;
}

bool ArrayIterator::Iter(void *x, void *y) {
    if (pos >= ndata)
        return false;

    int size = ndata - pos;
    if (size > bsz)
        size = bsz;

    UnpackData(
        xsrc, 
        xsrcDim0, 
        xsrcDim1, 
        xbuf, 
        xbufDim0, 
        xbufDim1, 
        size,
        false);
    if (ysrc != nullptr)
        UnpackData(
            ysrc, 
            ysrcDim0, 
            ysrcDim1, 
            ybuf, 
            ybufDim0, 
            ybufDim1, 
            size,
            makeOnehot);

    CudaMemcpyHtodAsync(x, xbuf, xbufDim0 * xbufDim1 * sizeof(float));
    if (ysrc != nullptr)
        CudaMemcpyHtodAsync(y, ybuf, ybufDim0 * ybufDim1 * sizeof(float));
    else
        CudaMemcpyHtodAsync(y, xbuf, xbufDim0 * xbufDim1 * sizeof(float));

    pos += bsz;
    if (size < bsz)
        start = bsz - size;

    return true;
}

// interface: testing

// TODO: Instututionalize this
bool ArrayIterator::TestIter(float *&x, float *&y) {
    if (pos >= ndata) {
        x = nullptr;
        y = nullptr;
        return false;
    }

    int size = ndata - pos;
    if (size > bsz)
        size = bsz;

    UnpackData(
        xsrc, 
        xsrcDim0, 
        xsrcDim1, 
        xbuf, 
        xbufDim0, 
        xbufDim1, 
        size,
        false);
    if (ysrc != nullptr)
        UnpackData(
            ysrc, 
            ysrcDim0, 
            ysrcDim1, 
            ybuf, 
            ybufDim0, 
            ybufDim1, 
            size,
            makeOnehot);

    x = xbuf;
    if (ysrc != nullptr)
        y = ybuf;
    else
        y = xbuf;

    pos += bsz;
    if (size < bsz)
        start = bsz - size;

    return true;
}

// implementation

void ArrayIterator::UnpackData(
        const float *in, 
        int inDim0, 
        int inDim1, 
        float *out,
        int outDim0,
        int outDim1,
        int size,
        bool onehot) {
    int inStart = pos;
    int inStop = pos + size;
    int outStart = 0;
    Unpack(
        in, 
        inDim0, 
        inDim1, 
        inStart, 
        inStop, 
        out, 
        outDim0, 
        outDim1, 
        outStart, 
        onehot);
    if (size < bsz) {
        // wrap around
        inStart = 0;
        inStop = bsz - size;
        outStart = size;
        Unpack(
            in, 
            inDim0, 
            inDim1, 
            inStart, 
            inStop, 
            out, 
            outDim0, 
            outDim1, 
            outStart, 
            onehot);
    }
}

void ArrayIterator::Unpack(
        const float *in, 
        int inDim0, 
        int inDim1, 
        int inStart, 
        int inStop,
        float *out,
        int outDim0,
        int outDim1,
        int outStart,
        bool onehot) {
    if (onehot)
        Onehot(
            in, 
            inDim0, 
            inDim1, 
            inStart, 
            inStop, 
            out, 
            outDim0, 
            outDim1, 
            outStart);
    else
        Transpose(
            in, 
            inDim0, 
            inDim1, 
            inStart, 
            inStop, 
            out, 
            outDim0, 
            outDim1, 
            outStart);
}

void ArrayIterator::Transpose(
        const float *in, 
        int inDim0, 
        int inDim1, 
        int inStart, 
        int inStop,
        float *out,
        int outDim0,
        int outDim1,
        int outStart) {
    assert(inDim1 == outDim0);
    assert(inStart < inDim0);
    if (inStop > inDim0)
        inStop = inDim0;
    // TODO: Optimize this loop?
    for (int i = inStart; i < inStop; i++) {
        for (int k = 0; k < inDim1; k++) {
            int inPos = i * inDim1 + k;
            int outPos = k * outDim1 + i - inStart + outStart;
            out[outPos] = in[inPos];
        }
    }
}

void ArrayIterator::Onehot(
        const float *in, 
        int inDim0, 
        int inDim1, 
        int inStart, 
        int inStop,
        float *out,
        int outDim0,
        int outDim1,
        int outStart) {
    assert(inDim1 == 1); // as I understand this (A.G.)
    assert(inStart < inDim0);
    if (inStop > inDim0)
        inStop = inDim0;
    for (int i = inStart; i < inStop; i++) {
        int hot = int(in[i]);
        for (int k = 0; k < outDim0; k++) {
            int outPos = k * outDim1 + i - inStart + outStart;
            out[outPos] = (k == hot) ? 1.0f : 0.0f;
        }
    }
}

} // cuda
} // arhat

