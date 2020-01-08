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

