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

#include "runtime/arhat.h"
#include "runtime_cuda/arhat.h"
#include "runtime_cuda/private.h"

namespace arhat {
namespace cuda {

//
//    Error handling
//

void CudaCheckError(cudaError_t stat, const char *file, int line) {
    if (stat != cudaSuccess)
        Error(cudaGetErrorString(stat), file, line);
}

} // cuda
} // arhat

