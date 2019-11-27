
#include <cuda_runtime.h>

#include "runtime_cuda/private.h"

namespace arhat {
namespace cuda {

//
//    Helper CUDA kernels
//

namespace {

//
//    Kernels
//

__global__ void KernelFill16(uint16_t *data, int count, uint16_t value) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < count) {
        data[tid] = value;
    }
}

__global__ void KernelFill32(uint32_t *data, int count, uint32_t value) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < count) {
        data[tid] = value;
    }
}

__global__ void KernelScale(float *data, int count, float alpha, float beta) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < count) {
        data[tid] = alpha * data[tid] + beta;
    }
}

} // namespace

//
//    Host
//

void Fill16(uint16_t *data, int count, uint16_t value) {
    int tpb = 256;
    int bpg = (count + tpb - 1) / tpb;
    KernelFill16<<<bpg, tpb>>>(data, count, value);
}

void Fill32(uint32_t *data, int count, uint32_t value) {
    int tpb = 256;
    int bpg = (count + tpb - 1) / tpb;
    KernelFill32<<<bpg, tpb>>>(data, count, value);
}

void Scale(float *data, int count, float alpha, float beta) {
    int tpb = 256;
    int bpg = (count + tpb - 1) / tpb;
    KernelScale<<<bpg, tpb>>>(data, count, alpha, beta);
}

} // cuda
} // arhat

