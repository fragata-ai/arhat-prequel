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
#include <cassert>

#include "runtime/arhat.h"

namespace arhat {

//
//    CannedImageLoaderBase
//

// construction/destruction

CannedImageLoaderBase::CannedImageLoaderBase() {
    xfp = nullptr;
    ndata = 0;
    height = 0;
    width = 0;
    nchan = 0;
    nclass = 0;
    makeOnehot = false;
    xsrc = nullptr;
    xsrcDim0 = 0;
    xsrcDim1 = 0;
    ysrc = nullptr;
    ysrcDim0 = 0;
    ysrcDim1 = 0;
    bsz = 0;
    xbuf = nullptr;
    xbufDim0 = 0;
    xbufDim1 = 0;
    ybuf = nullptr;
    ybufDim0 = 0;
    ybufDim1 = 0;
    start = 0;
    pos = 0;
    rgbMeanSubtract.enable = false;
    rgbMeanSubtract.pixelMean[0] = 0;
    rgbMeanSubtract.pixelMean[1] = 0;
    rgbMeanSubtract.pixelMean[2] = 0;
    valueNormalize.enable = false;
    valueNormalize.xmin = 0;
    valueNormalize.xspan = 0;
    valueNormalize.ymin = 0;
    valueNormalize.yspan = 0;
}

CannedImageLoaderBase::~CannedImageLoaderBase() {
    if (xfp != nullptr) {
        fclose(xfp);
    }
    delete[] xsrc;
    delete[] ysrc;
    delete[] xbuf;
    delete[] ybuf;
}

// interface

void CannedImageLoaderBase::Init(const char *xfname, const char *yfname, bool makeOnehot) {
    assert(xfp == nullptr);

    // images
    xfp = fopen(xfname, "rb");
    if (xfp == nullptr) {
        char buf[256];
        sprintf(buf, "Cannot open file [%s]\n", xfname);
        Error(buf, __FILE__, __LINE__);
    }
    uint32_t xhdr[4];
    if (fread(xhdr, sizeof(uint32_t), 4, xfp) != 4) {
        char buf[256];
        sprintf(buf, "Invalid header in [%s]\n", xfname);
        Error(buf, __FILE__, __LINE__);
    }
    ndata = int(xhdr[0]);
    height = int(xhdr[1]);
    width = int(xhdr[2]);
    nchan = int(xhdr[3]);

    // labels
    FILE *yfp = fopen(yfname, "rb");
    if (yfp == nullptr) {
        char buf[256];
        sprintf(buf, "Cannot open file [%s]\n", yfname);
        Error(buf, __FILE__, __LINE__);
    }
    uint32_t yhdr[2];
    if (fread(yhdr, sizeof(uint32_t), 2, yfp) != 2) {
        fclose(yfp);
        char buf[256];
        sprintf(buf, "Invalid header in [%s]\n", yfname);
        Error(buf, __FILE__, __LINE__);
    }
    if (xhdr[0] != yhdr[0]) {
        fclose(yfp);
        Error("Header data count mismatch", __FILE__, __LINE__);
    }
    nclass = int(yhdr[1]);

    ysrcDim0 = ndata;
    ysrcDim1 = 1;
    ysrc = new uint32_t[ysrcDim0 * ysrcDim1];
    if (fread(ysrc, sizeof(uint32_t), ndata, yfp) != ndata) {
        fclose(yfp);
        Error("Unexpected end of label file", __FILE__, __LINE__);
    }
    fclose(yfp);

    // validation
    if (makeOnehot) {
        for (int i = 0; i < ndata; i++) {
            if (!(ysrc[i] >= 0 && ysrc[i] < uint32_t(nclass))) {
                char buf[256];
                sprintf(buf, "Labels must range from 0 to %d", nclass - 1);
                Error(buf, __FILE__, __LINE__);
            }
        }
    }

    // other parameters
    this->makeOnehot = makeOnehot;

    xsrc = nullptr;
    xsrcDim0 = 0;
    xsrcDim1 = 0;

    bsz = 0;

    xbuf = nullptr;
    xbufDim0 = 0;
    xbufDim1 = 0;

    ybuf = nullptr;
    ybufDim0 = 0;
    ybufDim1 = 0;

    start = 0;
    pos = 0;

    rgbMeanSubtract.enable = false;
    valueNormalize.enable = false;
}

void CannedImageLoaderBase::RgbMeanSubtract(int rPixelMean, int gPixelMean, int bPixelMean) {
    if (nchan != 3) {
        char buf[256+1];
        sprintf(buf, "RgbMeanSubtract requires 3 channels, got %d", nchan);
        Error(buf, __FILE__, __LINE__);
    }
    int pixelMean[3];
    pixelMean[0] = rPixelMean;
    pixelMean[1] = gPixelMean;
    pixelMean[2] = bPixelMean;
    for (int i = 0; i < 3; i++) {
        if (pixelMean[i] < 0 || pixelMean[i] > 255) {
            char buf[256+1];
            sprintf(buf, "Mean pixel value out of range: %d", pixelMean[i]);
            Error(buf, __FILE__, __LINE__);
        }
    }
    rgbMeanSubtract.enable = true;
    rgbMeanSubtract.pixelMean[0] = uint8_t(pixelMean[0]);
    rgbMeanSubtract.pixelMean[1] = uint8_t(pixelMean[1]);
    rgbMeanSubtract.pixelMean[2] = uint8_t(pixelMean[2]);
}

void CannedImageLoaderBase::ValueNormalize(
        float sourceLow, float sourceHigh, float targetLow, float targetHigh) {
    valueNormalize.enable = true;
    valueNormalize.xmin = sourceLow;
    valueNormalize.xspan = sourceHigh - sourceLow;
    valueNormalize.ymin = targetLow;
    valueNormalize.yspan = targetHigh - targetLow;
}

void CannedImageLoaderBase::SetBsz(int bsz) {
    assert(this->bsz == 0);
    this->bsz = bsz;

    xsrcDim0 = bsz;
    xsrcDim1 = height * width * nchan;
    xsrc = new uint8_t[xsrcDim0 * xsrcDim1];

    xbufDim0 = xsrcDim1;
    xbufDim1 = bsz;
    xbuf = new float[xbufDim0 * xbufDim1];

    if (makeOnehot) {
        ybufDim0 = nclass;
    } else {
        ybufDim0 = 1;
    }
    ybufDim1 = bsz;
    ybuf = new float[ybufDim0 * ybufDim1];
}

int CannedImageLoaderBase::Nbatches() {
    assert(bsz != 0);
    return (ndata + bsz - 1) / bsz;
}

int CannedImageLoaderBase::Ndata() {
    return ndata;
}

void CannedImageLoaderBase::Reset() {
    start = 0;
    RewindImages();
}

void CannedImageLoaderBase::Start() {
    pos = start;
}

bool CannedImageLoaderBase::ReadBatch() {
    if (pos >= ndata) {
        return false;
    }

    int size = ndata - pos;
    if (size > bsz) {
        size = bsz;
    }

    UnpackImages(size);
    UnpackLabels(size);

    TransformImages();

    pos += bsz;
    if (size < bsz) {
        start = bsz - size;
    }

    return true;
}
 
float *CannedImageLoaderBase::XBuf() {
    return xbuf;
}

int CannedImageLoaderBase::XBufSize() {
    return xbufDim0 * xbufDim1;
}

float *CannedImageLoaderBase::YBuf() {
    return ybuf;
}

int CannedImageLoaderBase::YBufSize() {
    return ybufDim0 * ybufDim1;
}

// implementation

void CannedImageLoaderBase::UnpackImages(int size) {
    ReadImages(0, size);
    if (size < bsz) {
        // wrap around
        RewindImages();
        ReadImages(size, bsz - size);
    }
    UnpackU8(
        xsrc, 
        xsrcDim0, 
        xsrcDim1, 
        0, 
        bsz, 
        xbuf, 
        xbufDim0, 
        xbufDim1, 
        0);
    
}

void CannedImageLoaderBase::UnpackLabels(int size) {
    int inStart = pos;
    int inStop = pos + size;
    int outStart = 0;
    UnpackU32(
        ysrc, 
        ysrcDim0, 
        ysrcDim1, 
        inStart, 
        inStop, 
        ybuf, 
        ybufDim0, 
        ybufDim1, 
        outStart, 
        makeOnehot);
    if (size < bsz) {
        // wrap around
        inStart = 0;
        inStop = bsz - size;
        outStart = size;
        UnpackU32(
            ysrc, 
            ysrcDim0, 
            ysrcDim1, 
            inStart, 
            inStop, 
            ybuf, 
            ybufDim0, 
            ybufDim1, 
            outStart, 
            makeOnehot);
    }
}

void CannedImageLoaderBase::RewindImages() {
    // skip header
    if (fseek(xfp, long(4 * sizeof(uint32_t)), SEEK_SET) != 0) {
        Error("Image file rewind error", __FILE__, __LINE__);
    }
}

void CannedImageLoaderBase::ReadImages(int start, int size) {
    int offset = start * xsrcDim1;
    int count = size * xsrcDim1;
    if (fread(&xsrc[offset], sizeof(uint8_t), count, xfp) != count) {
        Error("Unexpected end of image file", __FILE__, __LINE__);
    }
}

void CannedImageLoaderBase::TransformImages() {
    // CHWN
    if (rgbMeanSubtract.enable) {
        int n = xbufDim0 * xbufDim1;
        int span = n / 3;
        for (int c = 0; c < 3; c++) {
            int start = c * span;
            int stop = start + span;
            float mean = float(rgbMeanSubtract.pixelMean[c]);
            for (int i = start; i < stop; i++)
                xbuf[i] -= mean;
        }
    }
    if (valueNormalize.enable) {
        int n = xbufDim0 * xbufDim1;
        // (x - xmin) / xspan * yspan + ymin = x * (yspan / xspan) + (ymin - xmin * yspan / xspan)
        float alpha = valueNormalize.yspan / valueNormalize.xspan;
        float beta = valueNormalize.ymin - valueNormalize.xmin * alpha;
        for (int i = 0; i < n; i++) {
            xbuf[i] = alpha * xbuf[i] + beta;
        }
    }
}

void CannedImageLoaderBase::UnpackU8(
        const uint8_t *in, 
        int inDim0, 
        int inDim1, 
        int inStart, 
        int inStop,
        float *out,
        int outDim0,
        int outDim1,
        int outStart) {
    // Transpose
    assert(inDim1 == outDim0);
    assert(inStart < inDim0);
    if (inStop > inDim0) {
        inStop = inDim0;
    }
    // TODO: Optimize this loop?
    for (int i = inStart; i < inStop; i++) {
        for (int k = 0; k < inDim1; k++) {
            int inPos = i * inDim1 + k;
            int outPos = k * outDim1 + i - inStart + outStart;
            out[outPos] = float(in[inPos]);
        }
    }
}

void CannedImageLoaderBase::UnpackU32(
        const uint32_t *in, 
        int inDim0, 
        int inDim1, 
        int inStart, 
        int inStop,
        float *out,
        int outDim0,
        int outDim1,
        int outStart,
        bool onehot) {
    if (onehot) {
        OnehotU32(
            in, 
            inDim0, 
            inDim1, 
            inStart, 
            inStop, 
            out, 
            outDim0, 
            outDim1, 
            outStart);
    } else {
        TransposeU32(
            in, 
            inDim0, 
            inDim1, 
            inStart, 
            inStop, 
            out, 
            outDim0, 
            outDim1, 
            outStart);
    }
}

void CannedImageLoaderBase::TransposeU32(
        const uint32_t *in, 
        int inDim0, 
        int inDim1, 
        int inStart, 
        int inStop,
        float *out,
        int outDim0,
        int outDim1,
        int outStart) {
    assert(inDim1 == outDim0);
    assert(inStart < inDim0);
    if (inStop > inDim0) {
        inStop = inDim0;
    }
    // TODO: Optimize this loop?
    for (int i = inStart; i < inStop; i++) {
        for (int k = 0; k < inDim1; k++) {
            int inPos = i * inDim1 + k;
            int outPos = k * outDim1 + i - inStart + outStart;
            out[outPos] = float(in[inPos]);
        }
    }
}

void CannedImageLoaderBase::OnehotU32(
        const uint32_t *in, 
        int inDim0, 
        int inDim1, 
        int inStart, 
        int inStop,
        float *out,
        int outDim0,
        int outDim1,
        int outStart) {
    assert(inDim1 == 1); // as I understand this (A.G.)
    assert(inStart < inDim0);
    if (inStop > inDim0) {
        inStop = inDim0;
    }
    for (int i = inStart; i < inStop; i++) {
        int hot = int(in[i]);
        for (int k = 0; k < outDim0; k++) {
            int outPos = k * outDim1 + i - inStart + outStart;
            out[outPos] = (k == hot) ? 1.0f : 0.0f;
        }
    }
}

} // arhat

