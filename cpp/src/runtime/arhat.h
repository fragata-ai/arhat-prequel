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
#include <cstdint>
#include <cfloat>
#include <cmath>

namespace arhat {

//
//    Common types
//

typedef unsigned char byte_t;

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
    DefaultSchedule();
    ~DefaultSchedule();
public:
    void Init(float change);
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
    void Init();
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
    PowerSchedule();
    ~PowerSchedule();
public:
    void Init(int step, float change);
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
//    Data iterators
//

class CannedImageLoaderBase {
public:
    CannedImageLoaderBase();
    ~CannedImageLoaderBase();
public:
    void Init(const char *xfname, const char *yfname, bool makeOnehot);
    void RgbMeanSubtract(int rPixelMean, int gPixelMean, int bPixelMean);
    void ValueNormalize(float sourceLow, float sourceHigh, float targetLow, float targetHigh); 
    void SetBsz(int bsz);
    int Nbatches();
    int Ndata();
    void Reset();
    void Start();
    bool ReadBatch();
    float *XBuf();
    int XBufSize();
    float *YBuf();
    int YBufSize();
private:
    void UnpackImages(int size);
    void UnpackLabels(int size);
    void RewindImages();
    void ReadImages(int start, int size);
    void TransformImages();
    static void UnpackU8(
        const uint8_t *in, 
        int inDim0, 
        int inDim1, 
        int inStart, 
        int inStop,
        float *out,
        int outDim0,
        int outDim1,
        int outStart);
    static void UnpackU32(
        const uint32_t *in, 
        int inDim0, 
        int inDim1, 
        int inStart, 
        int inStop,
        float *out,
        int outDim0,
        int outDim1,
        int outStart,
        bool onehot);
    static void TransposeU32(
        const uint32_t *in, 
        int inDim0, 
        int inDim1, 
        int inStart, 
        int inStop,
        float *out,
        int outDim0,
        int outDim1,
        int outStart);
    static void OnehotU32(
        const uint32_t *in, 
        int inDim0, 
        int inDim1, 
        int inStart, 
        int inStop,
        float *out,
        int outDim0,
        int outDim1,
        int outStart);
private:
    struct RgbMeanSubtractParam {
        bool enable;
        uint8_t pixelMean[3];
    };
    struct ValueNormalizeParam {
        bool enable;
        float xmin;
        float xspan;
        float ymin;
        float yspan;
    };
private:
    FILE *xfp;
    int ndata;
    int height;
    int width;
    int nchan;
    int nclass;
    bool makeOnehot;
    uint8_t *xsrc; // one batch
    int xsrcDim0;
    int xsrcDim1;
    uint32_t *ysrc; // all labels
    int ysrcDim0;
    int ysrcDim1;
    int bsz;
    float *xbuf;
    int xbufDim0;
    int xbufDim1;
    float *ybuf;
    int ybufDim0;
    int ybufDim1;
    int start;
    int pos;
    RgbMeanSubtractParam rgbMeanSubtract;
    ValueNormalizeParam valueNormalize;
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
    byte_t *Buffer(int index) const;
    void GetData(int index, void *buf) const;
    void Add(int size, const void *buf);
    void Load(const char *fname);
    void Save(const char *fname);
protected:
    void AddEntry(int size, byte_t *buf);
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
        byte_t *buf;
    };
private:
    int len;
    int cap;
    Entry *data;
};

} // arhat

