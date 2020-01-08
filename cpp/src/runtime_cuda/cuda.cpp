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
#include <cstdlib>
#include <cassert>

#include <cuda_runtime.h>

#include "runtime_cuda/arhat.h"
#include "runtime_cuda/private.h"

namespace arhat {
namespace cuda {

//
//    Global device memory
//

void *CudaMemAlloc(int bytes) {
    void *ptr = nullptr;
    cudaError_t stat = cudaMalloc(&ptr, bytes);
    CudaCheckError(stat, __FILE__, __LINE__);
    return ptr;
}

//
//    Initializing device memory
//

void CudaMemsetD8Async(void *dest, uint8_t data, int count) {
    cudaError_t stat = cudaMemsetAsync(dest, data, count);
    CudaCheckError(stat, __FILE__, __LINE__);
}

void CudaMemsetD16Async(void *dest, uint16_t data, int count) {
    if (data == 0) {
        cudaError_t stat = cudaMemsetAsync(dest, 0, 2 * count);
        CudaCheckError(stat, __FILE__, __LINE__);
    } else {
        Fill16((uint16_t *)dest, count, data);
    }
}

void CudaMemsetD32Async(void *dest, uint32_t data, int count) {
    if (data == 0) {
        cudaError_t stat = cudaMemsetAsync(dest, data, 4 * count);
        CudaCheckError(stat, __FILE__, __LINE__);
    } else {
        Fill32((uint32_t *)dest, count, data);
    }
}

//
//    Unstructured memory transfers
//

void CudaMemcpyHtod(void *dest, const void *src, int size) {
    cudaError_t stat = cudaMemcpy(dest, src, size, cudaMemcpyHostToDevice);
    CudaCheckError(stat, __FILE__, __LINE__);
}

void CudaMemcpyHtodAsync(void *dest, const void *src, int size) {
    cudaError_t stat = cudaMemcpyAsync(dest, src, size, cudaMemcpyHostToDevice);
    CudaCheckError(stat, __FILE__, __LINE__);
}

void CudaMemcpyDtoh(void *dest, const void *src, int size) {
    cudaError_t stat = cudaMemcpy(dest, src, size, cudaMemcpyDeviceToHost);
    CudaCheckError(stat, __FILE__, __LINE__);
}

void CudaMemcpyDtohAsync(void *dest, const void *src, int size) {
    cudaError_t stat = cudaMemcpyAsync(dest, src, size, cudaMemcpyDeviceToHost);
    CudaCheckError(stat, __FILE__, __LINE__);
}

void CudaMemcpyDtodAsync(void *dest, const void *src, int size) {
    cudaError_t stat = cudaMemcpyAsync(dest, src, size, cudaMemcpyDeviceToDevice);
    CudaCheckError(stat, __FILE__, __LINE__);
}

//
//    Structured memory transfer
//

float CudaGetFloat(void *src, int nbytes) {
    assert(nbytes == 4 || nbytes == 8);
    char buf[8];
    cudaError_t stat = cudaMemcpy(buf, src, nbytes, cudaMemcpyDeviceToHost);
    CudaCheckError(stat, __FILE__, __LINE__);
    float result = 0.0f;
    // ACHTUNG: Little endian only
    if (nbytes == 4)
        result = *reinterpret_cast<float *>(buf);
    else if (nbytes == 8)
        result = float(*reinterpret_cast<double *>(buf));
    return result;
}

//
//    Utility functions
//

// ACHTUNG: Implement CUDA kernel to perform reduction on device
float CudaGetFloatSum(float *data, int start, int stop) {
    int size = stop - start;
    if (size <= 0) {
        return 0.0f;
    }
    float *buf = new float[size];
    cudaError_t stat = cudaMemcpy(buf, &data[start], size * sizeof(float), cudaMemcpyDeviceToHost);
    if (stat != cudaSuccess) {
        delete[] buf;
        CudaCheckError(stat, __FILE__, __LINE__);
    }
    float result = 0.0f;
    for (int i = 0; i < size; i++)
        result += buf[i];
    delete[] buf;
    return result;
}

void CudaGetData(void *dest, int start, int stop, void *src, int itemSize) {
    if (start >= stop)
        return;
    char *p = (char *)dest + start * itemSize;
    int size = (stop - start) * itemSize;
    CudaMemcpyDtoh(p, src, size);
}

//
//    Memory management
//

void CudaMemGetInfo(size_t &free, size_t &total) {
    cudaMemGetInfo(&free, &total);
}

//
//    Events
//

//
//    CudaEvent
//

// construction/destruction

CudaEvent::CudaEvent() {
    cudaEventCreate(&handle);
}

CudaEvent::~CudaEvent() {
    cudaEventDestroy(handle);
}

// interface

void CudaEvent::Record() {
    cudaError_t stat = cudaEventRecord(handle, 0);
    CudaCheckError(stat, __FILE__, __LINE__);
}

void CudaEvent::Synchronize() {
    cudaError_t stat = cudaEventSynchronize(handle);
    CudaCheckError(stat, __FILE__, __LINE__);
}

double CudaEvent::TimeSince(CudaEvent &event) {
    float t;
    cudaError_t stat = cudaEventElapsedTime(&t, event.handle, handle);
    CudaCheckError(stat, __FILE__, __LINE__);
    return double(t);
}

} // cuda
} // arhat

