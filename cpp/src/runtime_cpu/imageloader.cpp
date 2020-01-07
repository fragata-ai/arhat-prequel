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
#include "runtime_cpu/arhat.h"

namespace arhat {
namespace cpu {

//
//    CannedImageLoader
//

// construction/destruction

CannedImageLoader::CannedImageLoader() { }

CannedImageLoader::~CannedImageLoader() { }

// interface

void CannedImageLoader::Init(const char *xfname, const char *yfname, bool makeOnehot) {
    base.Init(xfname, yfname, makeOnehot);
}

void CannedImageLoader::RgbMeanSubtract(int rPixelMean, int gPixelMean, int bPixelMean) {
    base.RgbMeanSubtract(rPixelMean, gPixelMean, bPixelMean);
}

void CannedImageLoader::ValueNormalize(
        float sourceLow, float sourceHigh, float targetLow, float targetHigh) {
    base.ValueNormalize(sourceLow, sourceHigh, targetLow, targetHigh);
}

// overrides

void CannedImageLoader::SetBsz(int bsz) {
    base.SetBsz(bsz);
}

int CannedImageLoader::Nbatches() {
    return base.Nbatches();
}

int CannedImageLoader::Ndata() {
    return base.Ndata();
}

void CannedImageLoader::Reset() {
    base.Reset();
}

void CannedImageLoader::Start() {
    base.Start();
}

bool CannedImageLoader::Iter(void *x, void *y) {
    if (!base.ReadBatch()) {
        return false;
    }
    CpuMemcpy(x, base.XBuf(), base.XBufSize() * sizeof(float));
    CpuMemcpy(y, base.YBuf(), base.YBufSize() * sizeof(float));
    return true;
}

} // cpu
} // arhat

