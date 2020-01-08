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

#include <cassert>
#include <cmath>

#include "runtime_cpu/arhat.h"

namespace arhat {
namespace cpu {

//
//    Convolution
//

/*
    N: Number of images in mini-batch
    C: Number of input feature maps
    K: Number of output feature maps

    D: Depth  of input image
    H: Height of input image
    W: Width  of input image

    T: Depth  of filter kernel
    R: Height of filter kernel
    S: Width  of filter kernel

    pad[D|H|W]: amount of zero-padding around the given edge
    str[D|H|W]: factor to step the filters by in a given direction
    dil[D|H|W]: dilation factor for each dimension
*/

namespace {

class FpropConvLayer {
public:
    FpropConvLayer(
        int D, 
        int H, 
        int W, 
        int T, 
        int R, 
        int S, 
        int M, 
        int P, 
        int Q,
        int strD,
        int strH,
        int strW,
        int padD,
        int padH,
        int padW,
        int dilD,
        int dilH,
        int dilW);
    ~FpropConvLayer();
public:
    int *MSlice(int index) const {
        return &mSlice[6*index];
    }
    int *PSlice(int index) const {
        return &pSlice[6*index];
    }
    int *QSlice(int index) const {
        return &qSlice[6*index];
    }
private:
    static void FpropSlice(
        int *slice, int q, int s, int x, int padding, int stride, int dilation);
private:
    int *buffer;
    int *mSlice;
    int *pSlice;
    int *qSlice;
};

FpropConvLayer::FpropConvLayer(
        int D, 
        int H, 
        int W, 
        int T, 
        int R, 
        int S, 
        int M, 
        int P, 
        int Q,
        int strD,
        int strH,
        int strW,
        int padD,
        int padH,
        int padW,
        int dilD,
        int dilH,
        int dilW) {
    buffer = new int[6*(M+P+Q)];
    int *slice = buffer;
    mSlice = slice;
    for (int im = 0; im < M; im++) {
        FpropSlice(slice, im, T, D, padD, strD, dilD);
        slice += 6;
    }
    pSlice = slice;
    for (int ip = 0; ip < P; ip++) {
        FpropSlice(slice, ip, R, H, padH, strH, dilH);
        slice += 6;
    }
    qSlice = slice;
    for  (int iq = 0; iq < Q; iq++) {
        FpropSlice(slice, iq, S, W, padW, strW, dilW);
        slice += 6;
    }
}

FpropConvLayer::~FpropConvLayer() {
    delete[] buffer;
}

void FpropConvLayer::FpropSlice(
        int *slice, int iq, int S, int X, int padding, int stride, int dilation) {
    int f1 = -1;
    int f2 = 0;
    int x1 = 0;
    int x2 = 0;
    int qs = iq * stride - padding;
    for (int is = 0; is < S; is++) {
        int ix = qs + is * dilation;
        if (ix >= 0 && ix < X) {
            if (f1 < 0) {
                x1 = ix;
                f1 = is;
            }
            x2 = ix;
            f2 = is;
        }
    }
    if (f1 < 0) {
        slice[0] = 0;
        slice[1] = 0;
        slice[2] = 1;
        slice[3] = 0;
        slice[4] = 0;
        slice[5] = 1;
        return;
    }
    slice[0] = f1;
    slice[1] = f2 + 1;
    slice[2] = 1;
    slice[3] = x1;
    slice[4] = x2 + 1;
    slice[5] = dilation;
}

class BpropConvLayer {
public:
    BpropConvLayer(
        int D, 
        int H, 
        int W, 
        int T, 
        int R, 
        int S, 
        int M, 
        int P, 
        int Q,
        int strD,
        int strH,
        int strW,
        int padD,
        int padH,
        int padW,
        int dilD,
        int dilH,
        int dilW);
    ~BpropConvLayer();
public:
    int *DSlice(int index) const {
        return &dSlice[6*index];
    }
    int *HSlice(int index) const {
        return &hSlice[6*index];
    }
    int *WSlice(int index) const {
        return &wSlice[6*index];
    }
private:
    static void BpropSlice(
        int *slice, int ix, int S, int Q, int padding, int stride, int dilation);
private:
    int *buffer;
    int *dSlice;
    int *hSlice;
    int *wSlice;
};

BpropConvLayer::BpropConvLayer(
        int D, 
        int H, 
        int W, 
        int T, 
        int R, 
        int S, 
        int M, 
        int P, 
        int Q,
        int strD,
        int strH,
        int strW,
        int padD,
        int padH,
        int padW,
        int dilD,
        int dilH,
        int dilW) {
    buffer = new int[6*(D+H+W)];
    int *slice = buffer;
    dSlice = slice;
    for (int id = 0; id < D; id++) {
        BpropSlice(slice, id, T, M, padD, strD, dilD);
        slice += 6;
    }
    hSlice = slice;
    for (int ih = 0; ih < H; ih++) {
        BpropSlice(slice, ih, R, P, padH, strH, dilH);
        slice += 6;
    }
    wSlice = slice;
    for (int iw = 0; iw < W; iw++) {
        BpropSlice(slice, iw, S, Q, padW, strW, dilW);
        slice += 6;
    }
}

BpropConvLayer::~BpropConvLayer() {
    delete[] buffer;
}

void BpropConvLayer::BpropSlice(
        int *slice, int ix, int S, int Q, int padding, int stride, int dilation) {
    int qs = ix - (dilation * (S - 1) - padding);
    int f1 = -1;
    int f2 = 0;
    int x1 = 0;
    int x2 = 0;
    for (int is = 0; is < S; is++) {
        int iq = qs + is * dilation;
        if (iq % stride == 0) {
            iq /= stride;
            if (iq >= 0 && iq < Q) {
                if (f1 < 0) {
                    f1 = is;
                    x1 = iq;
                }
                f2 = is;
                x2 = iq;
            }
        }
    }
    if  (f1 < 0) {
        slice[0] = 0;
        slice[1] = 0;
        slice[2] = 1;
        slice[3] = 0;
        slice[4] = 0;
        slice[5] = 1;
        return;
    }
    int fstep = 1;
    while ((fstep * dilation) % stride != 0) {
        fstep++;
    }
    int xstep = (fstep * dilation) / stride;
    slice[0] = f1;
    slice[1] = f2 + 1;
    slice[2] = fstep;
    slice[3] = x1;
    slice[4] = x2 + 1;
    slice[5] = xstep;
}

} // namespace

void FpropConv(
        float alpha, 
        float beta,
        const float *iData, // [C, D, H, W, N]
        const float *fData, // [C, T, R, S, K]
        float *oData,       // [K, M, P, Q, N]
        const float *xData, // [C, D, H, W, N]
        int C, 
        int D, 
        int H, 
        int W, 
        int N,
        int T, 
        int R, 
        int S, 
        int K,
        int M, 
        int P, 
        int Q,
        int strD,
        int strH, 
        int strW,
        int padD,
        int padH, 
        int padW,
        int dilD,
        int dilH, 
        int dilW) {
    FpropConvLayer layer(D, H, W, T, R, S, M, P, Q, 
        strD, strH, strW, padD, padH, padW, dilD, dilH, dilW);
    int WN = W * N;
    int HWN = H * WN;
    int DHWN = D * HWN;
    int SK = S * K;
    int RSK = R * SK;
    int TRSK = T * RSK;
    int QN = Q * N;
    int PQN = P * QN;
    int MPQN = M * PQN;
    for (int ik = 0; ik < K; ik++) {
    for (int im = 0; im < M; im++) {
    for (int ip = 0; ip < P; ip++) {
    for (int iq = 0; iq < Q; iq++) {
    for (int in = 0; in < N; in++) {
        int *mSlice = layer.MSlice(im);
        int *pSlice = layer.PSlice(ip);
        int *qSlice = layer.QSlice(iq);
        int tStart = mSlice[0];
        int tStop = mSlice[1];
        int tStep = mSlice[2];  // 1
        int dStart = mSlice[3];
        int dStep = mSlice[5];  // dilD
        int rStart = pSlice[0];
        int rStop = pSlice[1];
        int rStep = pSlice[2];  // 1
        int hStart = pSlice[3];
        int hStep = pSlice[5];  // dilH
        int sStart = qSlice[0];
        int sStop = qSlice[1];
        int sStep = qSlice[2];  // 1
        int wStart = qSlice[3];
        int wStep = qSlice[5];  // dilW
        float sum = 0.0f;
        for (int it = tStart, id = dStart; it < tStop; it += tStep, id += dStep) {
        for (int ir = rStart, ih = hStart; ir < rStop; ir += rStep, ih += hStep) {
        for (int is = sStart, iw = wStart; is < sStop; is += sStep, iw += wStep) {
            int fBase = RSK * it + SK * ir + K * is + ik;
            int iBase = HWN * id + WN * ih + N * iw + in;
            for (int ic = 0; ic < C; ic++) {
                int fIndex = TRSK * ic + fBase;
                int iIndex = DHWN * ic + iBase;
                sum += fData[fIndex] * iData[iIndex];
            }
        } // is
        } // ir
        } // it
        int oIndex = MPQN * ik + PQN * im + QN * ip + N * iq + in;
        sum *= alpha;
        if (beta != 0.0f) {
            sum += beta * xData[oIndex];
        }
        oData[oIndex] = sum;
    } // in
    } // iq
    } // ip
    } // im
    } // ik
}

void BpropConv(
        float alpha, 
        float beta,
        const float *iData, // E     [K, M, P, Q, N]
        const float *fData, // F     [C, T, R, S, K]
        float *oData,       // gradI [C, D, H, W, N]
        const float *xData, //       [K, M, P, Q, N]
        int C, 
        int D, 
        int H, 
        int W, 
        int N,
        int T, 
        int R, 
        int S, 
        int K,
        int M, 
        int P, 
        int Q,
        int strD,
        int strH, 
        int strW,
        int padD,
        int padH, 
        int padW,
        int dilD,
        int dilH, 
        int dilW) {
    BpropConvLayer layer(D, H, W, T, R, S, M, P, Q, 
        strD, strH, strW, padD, padH, padW, dilD, dilH, dilW);
    int WN = W * N;
    int HWN = H * WN;
    int DHWN = D * HWN;
    int SK = S * K;
    int RSK = R * SK;
    int TRSK = T * RSK;
    int QN = Q * N;
    int PQN = P * QN;
    int MPQN = M * PQN;
    for (int ic = 0; ic < C; ic++) {
    for (int id = 0; id < D; id++) {
    for (int ih = 0; ih < H; ih++) {
    for (int iw = 0; iw < W; iw++) {
    for (int in = 0; in < N; in++) {
        int *dSlice = layer.DSlice(id);
        int *hSlice = layer.HSlice(ih);
        int *wSlice = layer.WSlice(iw);
        int tStart = dSlice[0];
        int tStop = dSlice[1];
        int tStep = dSlice[2];
        int mStart = dSlice[3];
        int mStep = dSlice[5];
        int rStart = hSlice[0];
        int rStop = hSlice[1];
        int rStep = hSlice[2];
        int pStart = hSlice[3];
        int pStep = hSlice[5];
        int sStart = wSlice[0];
        int sStop = wSlice[1];
        int sStep = wSlice[2];
        int qStart = wSlice[3];
        int qStep = wSlice[5];
        float sum = 0.0f;
        for (int it = tStart, im = mStart; it < tStop; it += tStep, im += mStep) {
        for (int ir = rStart, ip = pStart; ir < rStop; ir += rStep, ip += pStep) {
        for (int is = sStart, iq = qStart; is < sStop; is += sStep, iq += qStep) {
            // F: C <=> K and mirror T, R, S (0, 1, 2, 3, 4) => (4, 1, 2, 3, 0)
            int fBase = TRSK * ic + RSK * (T - 1 - it) + SK * (R - 1 - ir) + K * (S - 1 - is);
            int iBase = PQN * im + QN * ip + N * iq + in;
            for (int ik = 0; ik < K; ik++) {
                int fIndex = ik + fBase;
                int iIndex = MPQN * ik + iBase;
                sum += fData[fIndex] * iData[iIndex];
            }
        } // is
        } // ir
        } // it
        int oindex = DHWN * ic + HWN * id + WN * ih + N * iw + in;
        sum *= alpha;
        if (beta != 0.0f) {
            sum += beta * xData[oindex];
        }
        oData[oindex] = sum;
    } // in
    } // iw
    } // ih
    } // id
    } // ic
}

void UpdateConv(
        float alpha,
        float beta,
        const float *iData, // [C, D, H, W, N]
        const float *eData, // [K, M, P, Q, N]
        float *uData,       // [C, T, R, S, K]
        int C, 
        int D, 
        int H, 
        int W, 
        int N,
        int T, 
        int R, 
        int S, 
        int K,
        int M, 
        int P, 
        int Q,
        int strD,
        int strH, 
        int strW,
        int padD,
        int padH, 
        int padW,
        int dilD,
        int dilH, 
        int dilW) {
    FpropConvLayer layer(D, H, W, T, R, S, M, P, Q, 
        strD, strH, strW, padD, padH, padW, dilD, dilH, dilW);
    int WN = W * N;
    int HWN = H * WN;
    int DHWN = D * HWN;
    int SK = S * K;
    int RSK = R * SK;
    int TRSK = T * RSK;
    int QN = Q * N;
    int PQN = P * QN;
    int MPQN = M * PQN;
    int uSize = C * TRSK;
    if (beta != 0.0f) {
        for (int uIndex = 0; uIndex < uSize; uIndex++) {
            uData[uIndex] *= beta;
        }
    } else {
        for (int uIndex = 0; uIndex < uSize; uIndex++) {
            uData[uIndex] = 0.0f;
        }
    }
    for (int ic = 0; ic < C; ic++) {
    for (int im = 0; im < M; im++) {
    for (int ip = 0; ip < P; ip++) {
    for (int iq = 0; iq < Q; iq++) {
    for (int ik = 0; ik < K; ik++) {
        int *mSlice = layer.MSlice(im);
        int *pSlice = layer.PSlice(ip);
        int *qSlice = layer.QSlice(iq);
        int tStart = mSlice[0];
        int tStop = mSlice[1];
        int tStep = mSlice[2];  // 1
        int dStart = mSlice[3];
        int dStep = mSlice[5];  // dilD
        int rStart = pSlice[0];
        int rStop = pSlice[1];
        int rStep = pSlice[2];  // 1
        int hStart = pSlice[3];
        int hStep = pSlice[5];  // dilH
        int sStart = qSlice[0];
        int sStop = qSlice[1];
        int sStep = qSlice[2];  // 1
        int wStart = qSlice[3];
        int wStep = qSlice[5];  // dilW
        for (int it = tStart, id = dStart; it < tStop; it += tStep, id += dStep) {
        for (int ir = rStart, ih = hStart; ir < rStop; ir += rStep, ih += hStep) {
        for (int is = sStart, iw = wStart; is < sStop; is += sStep, iw += wStep) {
            float sum = 0.0f;
            int eBase = MPQN * ik + PQN * im + QN * ip + N * iq;
            int iBase = DHWN * ic + HWN * id + WN * ih + N * iw;
            for (int in = 0; in < N; in++) {
                int eIndex = in + eBase;
                int iIndex = in + iBase;
                sum += eData[eIndex] * iData[iIndex];
            }
            int uIndex = TRSK * ic + RSK * it + SK * ir + K * is + ik;
            uData[uIndex] += alpha * sum;
        } // is
        } // ir
        } // it
    } // ik
    } // iq
    } // ip
    } // im
    } // ic
}

//
//    Pooling
//

/*
    op: max, avg, l2 pooling
    N: Number of images in mini-batch

    C: Number of input feature maps
    D: Depth  of input image
    H: Height of input image
    W: Width  of input image

    J: Size of feature map pooling window (maxout n_pieces)
    T: Depth  of pooling window
    R: Height of pooling window
    S: Width  of pooling window

    pad[C|D|H|W]: amount of zero-padding around the given image or feature map edge
    str[C|D|H|W]: factor to step the window by in a given direction (overlap allowed)
*/

namespace {

class PoolLayer {
public:
    PoolLayer(
        PoolOp op,
        int N,
        int C,
        int D,
        int H,
        int W,
        int J,
        int T,
        int R,
        int S,
        int K,
        int M,
        int P,
        int Q,
        int padC,
        int padD,
        int padH,
        int padW,
        int strC,
        int strD,
        int strH,
        int strW);
    ~PoolLayer();
public:
    int *KSlice(int index) const {
        return &kSlice[2*index];
    }
    int *MSlice(int index) const {
        return &mSlice[2*index];
    }
    int *PSlice(int index) const {
        return &pSlice[2*index];
    }
    int *QSlice(int index) const {
        return &qSlice[2*index];
    }
private:
    static void PoolSlice(int *slice, int iq, int S, int X, int padding, int strides);
private:
    int *buffer;
    int *kSlice;
    int *mSlice;
    int *pSlice;
    int *qSlice;
};

PoolLayer::PoolLayer(
        PoolOp op,
        int N,
        int C,
        int D,
        int H,
        int W,
        int J,
        int T,
        int R,
        int S,
        int K,
        int M,
        int P,
        int Q,
        int padC,
        int padD,
        int padH,
        int padW,
        int strC,
        int strD,
        int strH,
        int strW) {
    buffer = new int[2*(K+M+P+Q)];
    int *slice = buffer;
    kSlice = slice;
    for (int ik = 0; ik < K; ik++) {
        PoolSlice(slice, ik, J, C, padC, strC);
        slice += 2;
    }
    mSlice = slice;
    for (int im = 0; im < M; im++) {
        PoolSlice(slice, im, T, D, padD, strD);
        slice += 2;
    }
    pSlice = slice;
    for (int ip = 0; ip < P; ip++) {
        PoolSlice(slice, ip, R, H, padH, strH);
        slice += 2;
    }
    qSlice = slice;
    for (int iq = 0; iq < Q; iq++) {
        PoolSlice(slice, iq, S, W, padW, strW);
        slice += 2;
    }
}

PoolLayer::~PoolLayer() {
    delete[] buffer;
}

void PoolLayer::PoolSlice(int *slice, int iq, int S, int X, int padding, int strides) {
    int qs = iq * strides - padding;
    int firstI = -1;
    int lastI = 0;
    for (int is = 0; is < S; is++) {
        int ix = qs + is;
        if (ix >= 0 && ix < X) {
            if (firstI < 0) {
                firstI = ix;
            }
            lastI = ix;
        }
    }
    slice[0] = firstI;
    slice[1] = lastI + 1;
}

} // namespace

void FpropPool(
        const float *iData, // [C, D, H, W, N]
        float *oData,       // [K, M, P, Q, N]
        uint8_t *argmax,    // [K, M, P, Q, N]
        float beta,
        PoolOp op,
        int N,
        int C,
        int D,
        int H,
        int W,
        int J,
        int T,
        int R,
        int S,
        int K,
        int M,
        int P,
        int Q,
        int padC,
        int padD,
        int padH,
        int padW,
        int strC,
        int strD,
        int strH,
        int strW) {
    PoolLayer layer(op, N, C, D, H, W, J, T, R, S, K, M, P, Q,
        padC, padD, padH, padW, strC, strD, strH, strW);
    int WN = W * N;
    int HWN = H * WN;
    int DHWN = D * HWN;
    int QN = Q * N;
    int PQN = P * QN;
    int MPQN = M * PQN;
    int RS = R * S;
    int TRS = T * R * S;
    for (int ik = 0; ik < K; ik++) {
    for (int im = 0; im < M; im++) {
    for (int ip = 0; ip < P; ip++) {
    for (int iq = 0; iq < Q; iq++) {
        int *kSlice = layer.KSlice(ik);
        int *mSlice = layer.MSlice(im);
        int *pSlice = layer.PSlice(ip);
        int *qSlice = layer.QSlice(iq);
        int cStart = kSlice[0];
        int cStop = kSlice[1];
        int dStart = mSlice[0];
        int dStop = mSlice[1];
        int hStart = pSlice[0];
        int hStop = pSlice[1];
        int wStart = qSlice[0];
        int wStop = qSlice[1];
        int oBase = MPQN * ik + PQN * im + QN * ip + N * iq;
        if (op == PoolOp::Max)  {
            for (int in = 0;  in < N; in++) {
                int maxJ = 0;
                int maxT = 0;
                int maxR = 0;
                int maxS = 0;
                float maxval = -INFINITY;
                for (int ic = cStart; ic < cStop; ic++) {
                for (int id = dStart; id < dStop; id++) {
                for (int ih = hStart; ih < hStop; ih++) {
                for (int iw = wStart; iw < wStop; iw++) {
                    int iIndex = DHWN * ic + HWN * id + WN * ih + N * iw + in;
                    float val = iData[iIndex];
                    if (val > maxval) {
                        maxJ = ic - cStart;
                        maxT = id - dStart;
                        maxR = ih - hStart;
                        maxS = iw - wStart;
                        maxval = val;
                    }
                } // iw
                } // ih
                } // id
                } // ic
                int oIndex = oBase + in;
                int maxN = TRS * maxJ + RS * maxT + S * maxR + maxS;
                argmax[oIndex] = uint8_t(maxN);
                oData[oIndex] = oData[oIndex] * beta + maxval;
            } // in
        } else if (op == PoolOp::Avg) {
            for (int in = 0;  in < N; in++) {
                float sum = 0.0f;
                for (int ic = cStart; ic < cStop; ic++) {
                for (int id = dStart; id < dStop; id++) {
                for (int ih = hStart; ih < hStop; ih++) {
                for (int iw = wStart; iw < wStop; iw++) {
                    int iIndex = DHWN * ic + HWN * id + WN * ih + N * iw + in;
                    float val = iData[iIndex];
                    sum += val;
                } // iw
                } // ih
                } // id
                } // ic
                int count = (cStop - cStart) * (dStop - dStart) * (hStop - hStart) * (wStop - wStart);
                int oIndex = oBase + in;
                oData[oIndex] = oData[oIndex] * beta + sum / float(count);
            } // in
        } else if (op == PoolOp::L2) {
            for (int in = 0;  in < N; in++) {
                float sum = 0.0f;
                for (int ic = cStart; ic < cStop; ic++) {
                for (int id = dStart; id < dStop; id++) {
                for (int ih = hStart; ih < hStop; ih++) {
                for (int iw = wStart; iw < wStop; iw++) {
                    int iIndex = DHWN * ic + HWN * id + WN * ih + N * iw + in;
                    float val = iData[iIndex];
                    sum += val * val;
                } // iw
                } // ih
                } // id
                } // ic
                int count = (cStop - cStart) * (dStop - dStart) * (hStop - hStart) * (wStop - wStart);
                int oIndex = oBase + in;
                oData[oIndex] = oData[oIndex] * beta + std::sqrt(sum);
            } // in
        }
    } // iq
    } // ip
    } // im
    } // ik
}

void BpropPool(
        float *iData,           // [K, M, P, Q, N]
        float *oData,           // [C, D, H, W, N]
        const uint8_t *argmax,  // [K, M, P, Q, N]
        float alpha,
        float beta,
        PoolOp op,
        int N,
        int C,
        int D,
        int H,
        int W,
        int J,
        int T,
        int R,
        int S,
        int K,
        int M,
        int P,
        int Q,
        int padC,
        int padD,
        int padH,
        int padW,
        int strC,
        int strD,
        int strH,
        int strW) {
    PoolLayer layer(op, N, C, D, H, W, J, T, R, S, K, M, P, Q,
        padC, padD, padH, padW, strC, strD, strH, strW);
    int WN = W * N;
    int HWN = H * WN;
    int DHWN = D * HWN;
    int QN = Q * N;
    int PQN = P * QN;
    int MPQN = M * PQN;
    int RS = R * S;
    int TRS = T * R * S;
    if (alpha != 1.0f) {
        // ACHTUNG: Modifying input tensor: is this correct?
        int iSize = K * MPQN;
        for (int iIndex = 0; iIndex < iSize; iIndex++) {
            iData[iIndex] *= alpha;
        }
    }
    int oSize = C * DHWN;
    if (beta != 0.0f) {
        for (int oIndex = 0; oIndex < oSize; oIndex++) {
            oData[oIndex] *= beta;
        }
    } else {
        for (int oIndex = 0; oIndex < oSize; oIndex++) {
            oData[oIndex] = 0.0f;
        }
    }
    for (int ik = 0; ik < K; ik++) {
    for (int im = 0; im < M; im++) {
    for (int ip = 0; ip < P; ip++) {
    for (int iq = 0; iq < Q; iq++) {
        int *kSlice = layer.KSlice(ik);
        int *mSlice = layer.MSlice(im);
        int *pSlice = layer.PSlice(ip);
        int *qSlice = layer.QSlice(iq);
        int cStart = kSlice[0];
        int cStop = kSlice[1];
        int dStart = mSlice[0];
        int dStop = mSlice[1];
        int hStart = pSlice[0];
        int hStop = pSlice[1];
        int wStart = qSlice[0];
        int wStop = qSlice[1];
        if (op == PoolOp::Max)  {
            int iBase = MPQN * ik + PQN * im + QN * ip + N * iq;
            for (int in = 0; in < N; in++) {
                int iIndex = iBase + in;
                int maxN = argmax[iIndex];
                int ic = maxN / TRS + cStart;
                maxN %= TRS;
                int id = maxN / RS + dStart;
                maxN %= RS;
                int ih = maxN / S + hStart;
                maxN %= S;
                int iw = maxN + wStart;
                int oIndex = DHWN * ic + HWN * id + WN * ih + N * iw + in;
                float val = iData[iIndex];
                oData[oIndex] += val; 
            } // in
        } else if (op == PoolOp::Avg) {
            int count = (cStop - cStart) * (dStop - dStart) * (hStop - hStart) * (wStop - wStart);
            int iBase = MPQN * ik + PQN * im + QN * ip + N * iq;
            for (int in = 0;  in < N; in++) {
                int iIndex = iBase + in;
                float val = iData[iIndex] / float(count);
                for (int ic = cStart; ic < cStop; ic++) {
                for (int id = dStart; id < dStop; id++) {
                for (int ih = hStart; ih < hStop; ih++) {
                for (int iw = wStart; iw < wStop; iw++) {
                    int oIndex = DHWN * ic + HWN * id + WN * ih + N * iw + in;
                    oData[oIndex] += val;
                } // iw
                } // ih
                } // id
                } // ic
            } // in
        } else {
            // not implemented
            assert(false);
        }
    } // iq
    } // ip
    } // im
    } // ik
}

//
//    LRN
//

/*
    N: Number of images in mini-batch

    C: Number of input feature maps
    D: Depth  of input image 
    H: Height of input image
    W: Width  of input image

    J: Size of feature map pooling window (maxout n_pieces)
*/

namespace {

class LrnLayer: public PoolLayer {
public:
    LrnLayer(
        int N,
        int C,
        int D,
        int H,
        int W,
        int J);
    ~LrnLayer();
};

LrnLayer::LrnLayer(
        int N,
        int C,
        int D,
        int H,
        int W,
        int J):
    PoolLayer(
        PoolOp::Lrn,
        N, 
        C, 
        D, 
        H, 
        W, 
        J, 
        1,     // T
        1,     // R
        1,     // S
        C,     // K
        D,     // M
        H,     // P
        W,     // Q
        J / 2, // padC
        0,     // padD
        0,     // padH
        0,     // padW
        1,     // strC
        1,     // strD
        1,     // strH
        1) { } // strW

LrnLayer::~LrnLayer() { }

} // namespace

void FpropLrn(
        const float *iData, // [C, D, H, W, N]
        float *oData,       // [C, D, H, W, N]
        float *aData,       // [C, D, H, W, N]
        float ascale,
        float bpower,
        int N,
        int C,
        int D,
        int H,
        int W,
        int J) {
    LrnLayer layer(N, C, D, H, W, J);
    int K = C;
    int M = D;
    int P = H;
    int Q = W;
    int WN = W * N;
    int HWN = H * WN;
    int DHWN = D * HWN;
    int QN = Q * N;
    int PQN = P * QN;
    int MPQN = M * PQN;
    // although we can calculate directly into output oData,
    // keeping denom in aData around is useful for bprop
    ascale /= float(J);
    for (int ik = 0; ik < K; ik++) {
    for (int im = 0; im < M; im++) {
    for (int ip = 0; ip < P; ip++) {
    for (int iq = 0; iq < Q; iq++) {
        int *kSlice = layer.KSlice(ik);
        int *mSlice = layer.MSlice(im);
        int *pSlice = layer.PSlice(ip);
        int *qSlice = layer.QSlice(iq);
        int cStart = kSlice[0];
        int cStop = kSlice[1];
        int dStart = mSlice[0];
        int dStop = mSlice[1];
        int hStart = pSlice[0];
        int hStop = pSlice[1];
        int wStart = qSlice[0];
        int wStop = qSlice[1];
        int oBase = MPQN * ik + PQN * im + QN * ip + N * iq;
        for (int in = 0;  in < N; in++) {
            float sum = 0.0f;
            for (int ic = cStart; ic < cStop; ic++) {
            for (int id = dStart; id < dStop; id++) {
            for (int ih = hStart; ih < hStop; ih++) {
            for (int iw = wStart; iw < wStop; iw++) {
                int iIndex = DHWN * ic + HWN * id + WN * ih + N * iw + in;
                float val = iData[iIndex];
                sum += val * val;
            } // iw
            } // ih
            } // id
            } // ic
            int oIndex = oBase + in;
            aData[oIndex] = 1.0f + ascale * sum;
        } // in
    } // iq
    } // ip
    } // im
    } // ik
    int size = C * DHWN;
    for (int i = 0; i < size; i++) {
        // elementwise divide by denominator
        oData[i] = iData[i] * std::pow(aData[i], -bpower);
    }
}

void BpropLrn(
        const float *iData, // [C, D, H, W, N]
        const float *oData, // [C, D, H, W, N]
        const float *eData, // [C, D, H, W, N]
        float *dData,       // [C, D, H, W, N]
        const float *aData, // [C, D, H, W, N]
        float ascale,
        float bpower,
        int N,
        int C,
        int D,
        int H,
        int W,
        int J) {
    LrnLayer layer(N, C, D, H, W, J);
    int K = C;
    int M = D;
    int P = H;
    int Q = W;
    int WN = W * N;
    int HWN = H * WN;
    int DHWN = D * HWN;
    int QN = Q * N;
    int PQN = P * QN;
    int MPQN = M * PQN;
    for (int ik = 0; ik < K; ik++) {
    for (int im = 0; im < M; im++) {
    for (int ip = 0; ip < P; ip++) {
    for (int iq = 0; iq < Q; iq++) {
        int *kSlice = layer.KSlice(ik);
        int *mSlice = layer.MSlice(im);
        int *pSlice = layer.PSlice(ip);
        int *qSlice = layer.QSlice(iq);
        int cStart = kSlice[0];
        int cStop = kSlice[1];
        int dStart = mSlice[0];
        int dStop = mSlice[1];
        int hStart = pSlice[0];
        int hStop = pSlice[1];
        int wStart = qSlice[0];
        int wStop = qSlice[1];
        int oBase = MPQN * ik + PQN * im + QN * ip + N * iq;
        for (int in = 0;  in < N; in++) {
            float sum = 0.0f;
            for (int ic = cStart; ic < cStop; ic++) {
            for (int id = dStart; id < dStop; id++) {
            for (int ih = hStart; ih < hStop; ih++) {
            for (int iw = wStart; iw < wStop; iw++) {
                int iIndex = DHWN * ic + HWN * id + WN * ih + N * iw + in;
                float val = iData[iIndex];
                sum += oData[iIndex] * eData[iIndex] / aData[iIndex];
            } // iw
            } // ih
            } // id
            } // ic
            int oIndex = oBase + in;
            dData[oIndex] = sum;
        } // in
    } // iq
    } // ip
    } // im
    } // ik
    float coeff = -2.0f * bpower * (ascale / float(J));
    int size = C * DHWN;
    for (int i = 0; i < size; i++) {
        dData[i] = coeff * dData[i] * iData[i] + eData[i] * std::pow(aData[i], -bpower);
    }
}

} // cpu
} // arhat

