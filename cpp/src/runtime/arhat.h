#pragma once

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

#include <cstdio>
#include <cfloat>
#include <cmath>

namespace arhat {

//
//    Scalar math functions
//

inline bool IsFinite(float x) {
    return std::isfinite(x);
}

inline float Sgnf(float x) {
    return (x == 0.0f) ? 0.0f : copysignf(1.0f, x);
}

inline float Sqrf(float x) {
    return x * x;
}

inline float Safelogf(float x) {
    return (x > 0.0f) ? logf(x) : -50.0f;
}

inline float Sigf(float x) {
    return 1.0f / (1.0f + expf(-x));
}

inline float Sig2f(float x) {
    return 1.0f / (1.0f + exp2f(-x));
}

inline float Tanh2f(float x) {
    return (exp2f(2.0f * x) - 1.0f) / (exp2f(2.0f * x) + 1.0f);
}

//
//    Error handling
//

void Error(const char *msg, const char *file, int line);

//
//    Schedule
//

class Schedule {
public:
    Schedule() { }
    virtual ~Schedule() { }
public:
    virtual float GetLearningRate(float learningRate, int epoch) = 0;
};

//
//    DefaultSchedule
//

class DefaultSchedule: public Schedule {
public:
    DefaultSchedule(float change);
    ~DefaultSchedule();
public:
    void AddConfig(int step);
public:
    float GetLearningRate(float learningRate, int epoch);
private:
    enum { MAX_CONFIG = 1024 };
private:
    int stepConfig[MAX_CONFIG];
    float change;
    int numConfig;
    int steps;
};

//
//    StepSchedule
//

class StepSchedule: public Schedule {
public:
    StepSchedule();
    ~StepSchedule();
public:
    void AddConfig(int step, float change);
public:
    float GetLearningRate(float learningRate, int epoch);
private:
    enum { MAX_CONFIG = 1024 };
private:
    int stepConfig[MAX_CONFIG];
    float change[MAX_CONFIG];
    int numConfig;
    float steps;
};

//
//    PowerSchedule
//

class PowerSchedule: public Schedule {
public:
    PowerSchedule(int step, float change);
    ~PowerSchedule();
public:
    float GetLearningRate(float learningRate, int epoch);
private:
    int stepConfig;
    float change;
    int steps;
};

//
//    Callbacks
//

class Callbacks {
public:
    Callbacks();
    ~Callbacks();
public:
    void OnTrainBegin();
    void OnTrainEnd();
    void OnEpochBegin(int epoch);
    void OnEpochEnd(int epoch);
    void OnMinibatchBegin(int epoch, int minibatch);
    void OnMinibatchEnd(int epoch, int minibatch);
    bool Finished();
};

//
//    TODO: Callback interface and its implementations
//        (will require non-trivial design to call compound kernels)
//

//
//    Data sets
//

class Mnist {
public:
    Mnist();
    ~Mnist();
public:
    void Load(const char *path, bool normalize);
    int NumTrain() const
        { return numTrain; }
    int NumTest() const
        { return numTest; }
    int ImgSize() const
        { return imgSize; }
    float *XTrain() const
        { return xTrain; }
    float *YTrain() const
        { return yTrain; }
    float *XTest() const
        { return xTest; }
    float *YTest() const
        { return yTest; }
private:
    int numTrain;
    int numTest;
    int imgSize;
    float *xTrain;
    float *yTrain;
    float *xTest;
    float *yTest;
};

class Cifar10 {
public:
    Cifar10();
    ~Cifar10();
public:
    void Load(
        const char *path,
        bool normalize,
        bool contrastNormalize,
        bool whiten,
        bool padClasses);
    int NumTrain() const
        { return numTrain; }
    int NumTest() const
        { return numTest; }
    int ImgSize() const
        { return imgSize; }
    float *XTrain() const
        { return xTrain; }
    float *YTrain() const
        { return yTrain; }
    float *XTest() const
        { return xTest; }
    float *YTest() const
        { return yTest; }
private:
    void ReadData(FILE *fp, const char *tag, float *data, int count);
private:
    int numTrain;
    int numTest;
    int imgSize;
    float *xTrain;
    float *yTrain;
    float *xTest;
    float *yTest;
};

//
//    Array utilities
//

void TransposeSlice(void *dst, const void *src, int dim0, int dim1, int itemSize);

//
//    MemoryData
//

class MemoryData {
public:
    MemoryData();
    ~MemoryData();
public:
    void Reset();
    int Len() const;
    int Size(int index) const;
    void *Buffer(int index) const;
    void Add(int size, const void *buf);
    void Load(const char *fname);
    void Save(const char *fname);
protected:
    void AddEntry(int size, void *buf);
private:
    void Destroy();
    static int ReadInt(FILE *fp);
    static void WriteInt(FILE *fp, int val);
    static void ReadBuffer(FILE *fp, int size, void *buf);
    static void WriteBuffer(FILE *fp, int size, const void *buf);
    static void OpenError(const char *fname, const char *file, int line);
    static void EofError(const char *file, int line);
    static void IndexError(const char *file, int line);
private:
    struct Entry {
        int size;
        void *buf;
    };
private:
    int len;
    int cap;
    Entry *data;
};

} // arhat

