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

#include <cassert>
#include <cmath>

#include "runtime/arhat.h"

namespace arhat {

//
//    DefaultSchedule
//

// construction/destruction

DefaultSchedule::DefaultSchedule() {
    change = 0.0f;
    numConfig = 0;
    steps = 0;
}

DefaultSchedule::~DefaultSchedule() { }

// interface

void DefaultSchedule::Init(float change) {
    this->change = change;
    numConfig = 0;
    steps = 0;
}

void DefaultSchedule::AddConfig(int step) {
    assert(numConfig < MAX_CONFIG);
    stepConfig[numConfig] = step;
    numConfig++;
}

// overrides

float DefaultSchedule::GetLearningRate(float learningRate, int epoch) {
    steps = 0;
    for  (int i = 0; i < numConfig; i++) {
        if (stepConfig[i] <= epoch) {
            steps++;
        }
    }
    return learningRate * pow(change, float(steps));
}

//
//    StepSchedule
//

// construction/destruction

StepSchedule::StepSchedule() {
    numConfig = 0;
    steps = 0.0f;
}

StepSchedule::~StepSchedule() { }

// interface

void StepSchedule::Init() {
    numConfig = 0;
    steps = 0.0f;
}

void StepSchedule::AddConfig(int step, float change) {
    assert(numConfig < MAX_CONFIG);
    this->stepConfig[numConfig] = step;
    this->change[numConfig] = change;
    numConfig++;
}

// overrides

float StepSchedule::GetLearningRate(float learningRate, int epoch) {
    for (int i = 0; i < numConfig; i++) {
        if (stepConfig[i] == epoch) {
            steps = change[i];
            break;
        }
    }
    if (steps != 0.0f)
        return steps;
    return learningRate;
}

//
//    PowerSchedule
//

// construction/destruction

PowerSchedule::PowerSchedule() {
    stepConfig = 0;
    change = 0.0f;
    steps = 0;
}

PowerSchedule::~PowerSchedule() { }

// overrides

void PowerSchedule::Init(int step, float change) {
    this->stepConfig = step;
    this->change = change;
    steps = 0;
}

float PowerSchedule::GetLearningRate(float learningRate, int epoch) {
    steps = epoch / stepConfig;
    return learningRate * pow(change, float(steps));
}

} //arhat

