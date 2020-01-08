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

#include "runtime/arhat.h"

namespace arhat {

//
//    Callbacks
//

// construction/destruction

Callbacks::Callbacks() { 
    // TODO
}

Callbacks::~Callbacks() { }

// overrides

void Callbacks::OnTrainBegin() {
    printf("TrainBegin\n");
    // TODO
}

void Callbacks::OnTrainEnd() {
    printf("TrainEnd\n");
    // TODO
}

void Callbacks::OnEpochBegin(int epoch) {
//    printf("EpochBegin(%d)\n", epoch);
    // TODO
}

void Callbacks::OnEpochEnd(int epoch) {
//    printf("EpochEnd(%d)\n", epoch);
    // TODO
}

void Callbacks::OnMinibatchBegin(int epoch, int minibatch) {
//    printf("MinibatchBegin(%d, %d)\n", epoch, minibatch);
    // TODO
}

void Callbacks::OnMinibatchEnd(int epoch, int minibatch) {
//    printf("MinibatchEnd(%d, %d)\n", epoch, minibatch);
    // TODO
}

bool Callbacks::Finished() {
    return false;
}

} // arhat

