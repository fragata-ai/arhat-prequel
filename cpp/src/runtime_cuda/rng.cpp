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

#include <cuda_runtime.h>
#include <curand.h>

#include "runtime_cuda/arhat.h"
#include "runtime_cuda/private.h"

namespace arhat {
namespace cuda {

//
//    Random number generator
//

curandGenerator_t rng;

void RngInit() {
    curandStatus_t stat = curandCreateGenerator(&rng, CURAND_RNG_PSEUDO_DEFAULT);
    if (stat != CURAND_STATUS_SUCCESS) {
        fprintf(stderr, "RngInit failed\n");
        exit(1);
    }
}

void RngSetSeed(int seed) {
    curandStatus_t stat = curandSetPseudoRandomGeneratorSeed(rng, seed);
    if (stat != CURAND_STATUS_SUCCESS) {
        fprintf(stderr, "RngSetSeed failed\n");
        exit(1);
    }    
}

void RngNormal(float *out, float loc, float scale, int size) {
    curandStatus_t stat = curandGenerateNormal(rng, out, size, loc, scale);
    if (stat != CURAND_STATUS_SUCCESS) {
        fprintf(stderr, "RngNormal failed\n");
        exit(1);
    }
}

void RngUniform(float *out, float low, float high, int size) {
    curandStatus_t stat = curandGenerateUniform(rng, out, size);
    if (stat != CURAND_STATUS_SUCCESS) {
        fprintf(stderr, "RngUniform failed\n");
        exit(1);
    }
    if (low == 0.0f && high == 1.0f)
        return;
    // Transform [0, 1) to [low, high)
    Scale(out, size, high - low, low);
}

} // cuda
} // arhat

