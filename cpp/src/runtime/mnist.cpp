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
#include <cstdint>
#include <string>

#include "runtime/arhat.h"

namespace arhat {

//
//    Mnist
//

// construction/destruction

Mnist::Mnist() {
    numTrain = 0;
    numTest = 0;
    imgSize = 0;
    xTrain = nullptr;
    yTrain = nullptr;
    xTest = nullptr;
    yTest = nullptr;
}

Mnist::~Mnist() {
    delete[] xTrain;
    delete[] yTrain;
    delete[] xTest;
    delete[] yTest;
}

// iterface

void Mnist::Load(const char *path, bool normalize) {
    std::string fn = std::string(path) + "/mnist.dat";

    FILE *fp = fopen(fn.c_str(), "rb");
    if (fp == nullptr) {
        char buf[256];
        sprintf(buf, "Cannot open file [%s]\n", fn.c_str());
        Error(buf, __FILE__, __LINE__);
    }

    numTrain = 60000;
    numTest = 10000;
    imgSize = 28;

    int xTrainSize = numTrain * imgSize * imgSize;
    int yTrainSize = numTrain;
    int xTestSize = numTest * imgSize * imgSize;
    int yTestSize = numTest;

    int dataSize = xTrainSize + yTrainSize + xTestSize + yTestSize;
    uint8_t *data = new uint8_t[dataSize];
    int nread = int(fread(data, 1, dataSize, fp));
    fclose(fp);
    if (nread != dataSize) {
        delete[] data;
        char buf[256];
        sprintf(buf, "Data set too short: want %d, got %d\n", dataSize, nread);
        Error(buf, __FILE__, __LINE__);
    }

    xTrain = new float[xTrainSize];
    yTrain = new float[yTrainSize];
    xTest = new float[xTestSize];
    yTest = new float[yTestSize];

    int k = 0;
    for (int i = 0; i < xTrainSize; i++) {
        xTrain[i] = float(data[k]);
        k++;
    }
    for (int i = 0; i < yTrainSize; i++) {
        yTrain[i] = float(data[k]);
        k++;
    }
    for (int i = 0; i < xTestSize; i++) {
        xTest[i] = float(data[k]);
        k++;
    }
    for (int i = 0; i < yTestSize; i++) {
        yTest[i] = float(data[k]);
        k++;
    }

    if (normalize) {
        for (int i = 0; i < xTrainSize; i++) {
            xTrain[i] /= 255.0f;
        }
        for (int i = 0; i < xTestSize; i++) {
            xTest[i] /= 255.0f;
        }
    }

    delete[] data;
}

} // arhat

