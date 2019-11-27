//
// Copyright 2019 FRAGATA COMPUTER SYSTEMS AG
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
#include <cstdint>
#include <string>

#include "runtime/arhat.h"

namespace arhat {

//
//    Cifar10
//

// construction/destruction

Cifar10::Cifar10() {
    numTrain = 0;
    numTest = 0;
    imgSize = 0;
    xTrain = nullptr;
    yTrain = nullptr;
    xTest = nullptr;
    yTest = nullptr;
}

Cifar10::~Cifar10() {
    delete[] xTrain;
    delete[] yTrain;
    delete[] xTest;
    delete[] yTest;
}

// iterface

void Cifar10::Load(
        const char *path,
        bool normalize,
        bool contrastNormalize,
        bool whiten,
        bool padClasses) {
    // Whitening is not yet implemented; prefabricated data sets are used instead
    // File naming conventions: cifar10[_suffix].dat where suffix contains
    // n = normalize, c = contrastNormalize, w = whiten
    std::string fn = std::string(path) + "/cifar10";
    std::string suffix("_");
    if (normalize) {
        suffix += 'n';
    }
    if (contrastNormalize) {
        suffix += 'c';
    }
    if (whiten) {
        suffix += 'w';
    }
    if (suffix.length() > 1) {
        fn += suffix;
    }
    fn += ".dat";

    FILE *fp = fopen(fn.c_str(), "rb");
    if (fp == nullptr) {
        char buf[256];
        sprintf(buf, "Cannot open file [%s]\n", fn.c_str());
        Error(buf, __FILE__, __LINE__);
    }

    numTrain = 50000;
    numTest = 10000;
    imgSize = 32;

    int xTrainSize = numTrain * 3 * imgSize * imgSize;
    int yTrainSize = numTrain;
    int xTestSize = numTest * 3 * imgSize * imgSize;
    int yTestSize = numTest;

    xTrain = new float[xTrainSize];
    yTrain = new float[yTrainSize];
    xTest = new float[xTestSize];
    yTest = new float[yTestSize];

    ReadData(fp, "xTrain", xTrain, xTrainSize);
    ReadData(fp, "yTrain", yTrain, yTrainSize);
    ReadData(fp, "xTest", xTest, xTestSize);
    ReadData(fp, "yTest", yTest, yTestSize);
}

// implementation

void Cifar10::ReadData(FILE *fp, const char *tag, float *data, int count) {
    int nread = int(fread(data, sizeof(float), count, fp));
    if (nread != count) {
        char buf[256];
        sprintf(buf, "Data set %s too short: want %d, got %d\n", tag, count, nread);
        Error(buf, __FILE__, __LINE__);
    }
}

} // arhat

