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
#include <cstdlib>

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "runtime_cuda/arhat.h"

namespace arhat {
namespace cuda {

//
//    CUBLAS
//

cublasHandle_t cublasHandle;

void CublasInit() {
    cublasStatus_t stat = cublasCreate(&cublasHandle);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "CublasInit failed\n");
        exit(1);
    }
}

void CublasSgemm(
        int transa, // 'n'/'t'
        int transb, // 'n'/'t'
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
        int ldc) {
    cublasStatus_t stat =
        cublasSgemm(
             cublasHandle,
             (transa == 't') ? CUBLAS_OP_T : CUBLAS_OP_N,
             (transb == 't') ? CUBLAS_OP_T : CUBLAS_OP_N,
             m,
             n,
             k,
             &alpha,
             a,
             lda,
             b,
             ldb,
             &beta,
             c,
             ldc);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "CublasSgemm failed\n");
        exit(1);
    }
}

void CublasSgemv(
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
    int incy) {
    cublasStatus_t stat =
        cublasSgemv(
             cublasHandle,
             (trans == 't') ? CUBLAS_OP_T : CUBLAS_OP_N,
             m,
             n,
             &alpha,
             a,
             lda,
             x,
             incx,
             &beta,
             y,
             incy);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "CublasSgemv failed\n");
        exit(1);
    }
}

} // cuda
} // arhat

