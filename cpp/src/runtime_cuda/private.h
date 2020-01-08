#pragma once

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

#include <cstdint>

namespace arhat {
namespace cuda {

//
//    Error handling
//

void CudaCheckError(cudaError_t stat, const char *file, int line);

//
//    Helper CUDA kernels
//

void Fill16(uint16_t *data, int count, uint16_t value);
void Fill32(uint32_t *data, int count, uint32_t value);

void Scale(float *data, int count, float alpha, float beta);

} // cuda
} // arhat

