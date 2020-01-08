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

#include "runtime/arhat.h"
#include "runtime_cpu/arhat.h"

#include "private.h"

#include <random>

namespace arhat {
namespace cpu {

namespace {

std::default_random_engine rng;

} // namespace

void RngInit() {
    // Nothing to do
}

void RngSetSeed(int seed) {
    rng.seed(seed);
}

void RngNormal(float *out, float loc, float scale, int size) {
    std::normal_distribution<float> dist(loc, scale);
    for (int i = 0; i < size; i++)
        out[i] = dist(rng);
}

void RngUniform(float *out, float low, float high, int size) {
    std::uniform_real_distribution<float> dist(low, high);
    for (int i = 0; i < size; i++)
        out[i] = dist(rng);
}

// used internally by kernels

float Frand() {
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    return dist(rng);
}

} // cpu
} // arhat

