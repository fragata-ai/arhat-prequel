#pragma once

//
// Copyright 2019 FRAGATA COMPUTER SYSTEMS AG
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

#include "runtime/arhat.h"

namespace arhat {
namespace cpu {

//
//    ---- Platform primitives
//

//
//    Global device memory
//

void *CpuMemAlloc(int bytes);

//
//    Initializing device memory
//

void CpuMemsetD8(void *dest, uint8_t data, int count);
void CpuMemsetD16(void *dest, uint16_t data, int count);
void CpuMemsetD32(void *dest, uint32_t data, int count);

//
//    Unstructured memory transfers
//

void CpuMemcpy(void *dest, const void *src, int size);

//
//    Structured memory transfer
//

float CpuGetFloat(void *src, int nbytes);

//
//    Utility functions
//

float CpuGetFloatSum(float *src, int start, int stop);
void CpuGetData(void *dest, int start, int stop, void *src, int itemSize);

//
//    Random number generator
//

void RngInit();
void RngSetSeed(int seed);
void RngNormal(float *out, float loc, float scale, int size);
void RngUniform(float *out, float low, float high, int size);

//
//    BLAS routines
//

int BlasSgemm(
        int transa, 
        int transb, 
        int m, 
        int n, 
        int k, 
        float alpha, 
        const float *a, 
        int lda, 
        const float *b, 
        int ldb, 
        float beta, 
        float *c, 
        int ldc);

int BlasSgemv(
        int trans, 
        int m, 
        int n, 
        float alpha,
        const float *a, 
        int lda, 
        const float *x, 
        int incx, 
        float beta, 
        float *y,
        int incy);

//
//    DNN routines
//

void FpropConv(
        float alpha, 
        float beta,
        const float *iData, // [C, D, H, W, N]
        const float *fData, // [C, T, R, S, K]
        float *oData,       // [K, M, P, Q, N]
        const float *xData, // [C, D, H, W, N]
        int C, 
        int D, 
        int H, 
        int W, 
        int N,
        int T, 
        int R, 
        int S, 
        int K,
        int M, 
        int P, 
        int Q,
        int strD,
        int strH, 
        int strW,
        int padD,
        int padH, 
        int padW,
        int dilD,
        int dilH, 
        int dilW);

void BpropConv(
        float alpha, 
        float beta,
        const float *iData, // E     [K, M, P, Q, N]
        const float *fData, // F     [C, T, R, S, K]
        float *oData,       // gradI [C, D, H, W, N]
        const float *xData, //       [K, M, P, Q, N]
        int C, 
        int D, 
        int H, 
        int W, 
        int N,
        int T, 
        int R, 
        int S, 
        int K,
        int M, 
        int P, 
        int Q,
        int strD,
        int strH, 
        int strW,
        int padD,
        int padH, 
        int padW,
        int dilD,
        int dilH, 
        int dilW);

void UpdateConv(
        float alpha,
        float beta,
        const float *iData, // [C, D, H, W, N]
        const float *eData, // [K, M, P, Q, N]
        float *uData,       // [C, T, R, S, K]
        int C, 
        int D, 
        int H, 
        int W, 
        int N,
        int T, 
        int R, 
        int S, 
        int K,
        int M, 
        int P, 
        int Q,
        int strD,
        int strH, 
        int strW,
        int padD,
        int padH, 
        int padW,
        int dilD,
        int dilH, 
        int dilW);

enum class PoolOp {
    Max,
    Avg,
    L2,
    Lrn
};

void FpropPool(
        const float *iData, // [C, D, H, W, N]
        float *oData,       // [K, M, P, Q, N]
        uint8_t *argmax,    // [K, M, P, Q, N]
        float beta,
        PoolOp op,
        int N,
        int C,
        int D,
        int H,
        int W,
        int J,
        int T,
        int R,
        int S,
        int K,
        int M,
        int P,
        int Q,
        int padC,
        int padD,
        int padH,
        int padW,
        int strC,
        int strD,
        int strH,
        int strW);

void BpropPool(
        float *iData,          // [K, M, P, Q, N]
        float *oData,          // [C, D, H, W, N]
        const uint8_t *argmax, // [K, M, P, Q, N]
        float alpha,
        float beta,
        PoolOp op,
        int N,
        int C,
        int D,
        int H,
        int W,
        int J,
        int T,
        int R,
        int S,
        int K,
        int M,
        int P,
        int Q,
        int padC,
        int padD,
        int padH,
        int padW,
        int strC,
        int strD,
        int strH,
        int strW);

void FpropLrn(
        const float *iData, // [C, D, H, W, N]
        float *oData,       // [C, D, H, W, N]
        float *aData,       // [C, D, H, W, N]
        float ascale,
        float bpower,
        int N,
        int C,
        int D,
        int H,
        int W,
        int J);

void BpropLrn(
        const float *iData, // [C, D, H, W, N]
        const float *oData, // [C, D, H, W, N]
        const float *eData, // [C, D, H, W, N]
        float *dData,       // [C, D, H, W, N]
        const float *aData, // [C, D, H, W, N]
        float ascale,
        float bpower,
        int N,
        int C,
        int D,
        int H,
        int W,
        int J);

//
//    ---- Arhat
//

//
//    DataIterator
//

class DataIterator {
public:
    DataIterator() { }
    virtual ~DataIterator() { }
public:
    virtual void SetBsz(int bsz) = 0;
    virtual int Nbatches() = 0;
    virtual int Ndata() = 0;
    virtual void Reset() = 0;
    virtual void Start() = 0;
    virtual bool Iter(void *x, void *y) = 0;
};

class ArrayIterator: public DataIterator {
public:
    ArrayIterator();
    ~ArrayIterator();
public:
    void Init(
        float *x,
        int xdim0,
        int xdim1,
        float *y,
        int ydim1,
        int nclass,
        bool makeOnehot);
public:
    void SetBsz(int bsz);
    int Nbatches();
    int Ndata();
    void Reset();
    void Start();
    bool Iter(void *x, void *y);
public:
    bool TestIter(float *&x, float *&y);
private:
    void UnpackData(
        const float *in, 
        int inDim0, 
        int inDim1, 
        float *out,
        int outDim0,
        int outDim1,
        int size,
        bool onehot);
    static void Unpack(
        const float *in, 
        int inDim0, 
        int inDim1, 
        int inStart, 
        int inStop,
        float *out,
        int outDim0,
        int outDim1,
        int outStart,
        bool onehot);
    static void Transpose(
        const float *in, 
        int inDim0, 
        int inDim1, 
        int inStart, 
        int inStop,
        float *out,
        int outDim0,
        int outDim1,
        int outStart);
    static void Onehot(
        const float *in, 
        int inDim0, 
        int inDim1, 
        int inStart, 
        int inStop,
        float *out,
        int outDim0,
        int outDim1,
        int outStart);
private:
    int ndata;
    int start;
    int nclass;
    bool makeOnehot;
    float *xsrc;
    int xsrcDim0;
    int xsrcDim1;
    float *ysrc;
    int ysrcDim0;
    int ysrcDim1;
    int bsz;
    float *xbuf;
    int xbufDim0;
    int xbufDim1;
    float *ybuf;
    int ybufDim0;
    int ybufDim1;
    int pos;
};

//
//    ArrayWriter
//

class DataWriter {
public:
    DataWriter() { }
    virtual ~DataWriter() { }
public:
    virtual void WriteBatch(const void *buf, int count) = 0;
};

class ArrayWriter: public DataWriter {
public:
    ArrayWriter();
    ~ArrayWriter();
public:
    void Init(int dim1, int bsz, int itemSize);
    int Len() const;
    int Size(int index) const;
    byte_t *Buffer(int index) const;
    void GetData(int index, void *buf) const;
public:
    void WriteBatch(const void *buf, int count);
private:
    MemoryData data;
    int dim1;
    int bsz;
    int itemSize;
    byte_t *xfer0;
    byte_t *xfer1;
};

//
//    MemoryReader
//

class MemoryReader: public MemoryData {
public:
    MemoryReader();
    ~MemoryReader();
public:
    void Start();
    void Read(int size, void *buf);
private:
    int pos;
};

//
//    MemoryWriter
//

class MemoryWriter: public MemoryData {
public:
    MemoryWriter();
    ~MemoryWriter();
public:
    void Write(int size, const void *buf);
};

} // cpu
} // arhat

