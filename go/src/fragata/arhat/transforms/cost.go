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
// Ported from Python to Go and partly modified by FRAGATA COMPUTER SYSTEMS AG.
//

package transforms

import (
    "fmt"
    "fragata/arhat/backends"
    "fragata/arhat/base"
    "math"
)

//
//    local functions
//

func assertSingle(x []backends.Tensor) {
    base.Assert(len(x) == 1)
}

//
//    Cost
//

type Cost interface {
    base.Object
    Call(y backends.Value, t backends.Value) backends.Value
    Bprop(y backends.Value, t backends.Value) backends.Value
}

//
//    CostArg
//

type CostArg struct {
    base.ArgDefBase
}

func NewCostArg() *CostArg {
    a := new(CostArg)
    a.ArgDefBase.Init(true, nil)
    return a
}

func NewCostArgOpt(defval Cost) *CostArg {
    a := new(CostArg)
    a.ArgDefBase.Init(false, defval)
    return a
}

func(a *CostArg) Expand(v interface{}) (interface{}, bool) {
    if v == nil {
        return v, true
    }
    if _, ok := v.(Cost); !ok {
        return nil, false
    }
    return v, true
}

func ToCost(v interface{}) Cost {
    if v == nil {
        return nil
    }
    return v.(Cost)
}

//
//    CostBase
//

type CostBase struct {
    base.ObjectBase
}

func(c *CostBase) Init(self base.Object) {
    c.ObjectBase.Init(self, nil)
}

//
//    CrossEntropyBinary
//

type CrossEntropyBinary struct {
    CostBase
    scale float64
}

var crossEntropyBinaryInitArgMap = base.ArgMap{
    "scale": base.NewFloatArgOpt(1.0),
}

func NewCrossEntropyBinary(args ...interface{}) *CrossEntropyBinary {
    c := new(CrossEntropyBinary)
    c.Init(c, base.MakeArgs(args))
    return c
}

func(c *CrossEntropyBinary) Init(self base.Object, args base.Args) {
    args = crossEntropyBinaryInitArgMap.Expand(args)
    c.CostBase.Init(self)
    c.scale = args["scale"].(float64)
}

func(c *CrossEntropyBinary) Call(y backends.Value, t backends.Value) backends.Value {
    // sum(safelog(1 - y) * (t - 1) - safelog(y) * t, axis=0)
    base.AssertMsg(base.IntsEq(y.Shape(), t.Shape()), 
        "CrossEntropy requires network output shape to match targets")
    be := backends.Be()
    one := be.Float(1.0);
    return be.Sum(be.Safelog(one.Sub(y)).Mul(t.Sub(one)).Sub(be.Safelog(y).Mul(t)), 0)
}

func(c *CrossEntropyBinary) Bprop(y backends.Value, t backends.Value) backends.Value {
    // scale * (y - t)
    be := backends.Be()
    scale := be.Float(c.scale);
    return scale.Mul(y.Sub(t))
}

func(c *CrossEntropyBinary) ClassName() string {
    return "arhat.costs.CrossEntropyBinary"
}

//
//    CrossEntropyMulti
//

type CrossEntropyMulti struct {
    CostBase
    usebits bool
    scale float64
    logscale float64
}

var crossEntropyMultiInitArgMap = base.ArgMap{
    "scale": base.NewFloatArgOpt(1.0),
    "usebits": base.NewBoolArgOpt(false),
}

func NewCrossEntropyMulti(args ...interface{}) *CrossEntropyMulti {
    c := new(CrossEntropyMulti)
    c.Init(c, base.MakeArgs(args))
    return c
}

func(c *CrossEntropyMulti) Init(self base.Object, args base.Args) {
    args = crossEntropyMultiInitArgMap.Expand(args)
    c.CostBase.Init(self)
    c.usebits = args["usebits"].(bool)
    c.scale = args["scale"].(float64)
    if c.usebits {
        c.logscale = 1.0 / math.Log(2.0)
    } else {
        c.logscale = 1.0
    }
}

func(c *CrossEntropyMulti) Call(y backends.Value, t backends.Value) backends.Value {
    // sum(-t * logscale * safelog(y), axis=0)
    base.AssertMsg(base.IntsEq(y.Shape(), t.Shape()), 
        "CrossEntropy requires network output shape to match targets")
    be := backends.Be()
    logscale := be.Float(c.logscale)
    return be.Sum(t.Neg().Mul(logscale).Mul(be.Safelog(y)), 0)
}

func(c *CrossEntropyMulti) Bprop(y backends.Value, t backends.Value) backends.Value {
    // logscale * scale * (y - t)
    be := backends.Be()
    scale := be.Float(c.scale);
    logscale := be.Float(c.logscale)
    return logscale.Mul(scale).Mul(y.Sub(t))
}

func(c *CrossEntropyMulti) ClassName() string {
    return "arhat.costs.CrossEntropyMulti"
}

//
//    SumSquared
//

type SumSquared struct {
    CostBase
}

func NewSumSquared() *SumSquared {
    c := new(SumSquared)
    c.Init(c)
    return c
}

func(c *SumSquared) Init(self base.Object) {
    c.CostBase.Init(self)
}

func(c *SumSquared) Call(y backends.Value, t backends.Value) backends.Value {
    // sum(square(y - t), axis=0) / 2.0
    be := backends.Be()
    half := be.Float(0.5);
    return be.Sum(be.Square(y.Sub(t)), 0).Mul(half)
}

func(c *SumSquared) Bprop(y backends.Value, t backends.Value) backends.Value {
    // y - t
    return y.Sub(t)
}

func(c *SumSquared) ClassName() string {
    return "arhat.costs.SumSquared"
}

//
//    MeanSquared
//

type MeanSquared struct {
    CostBase
}

func NewMeanSquared() *MeanSquared {
    c := new(MeanSquared)
    c.Init(c)
    return c
}

func(c *MeanSquared) Init(self base.Object) {
    c.CostBase.Init(self)
}

func(c *MeanSquared) Call(y backends.Value, t backends.Value) backends.Value {
    // mean(square(y - t), axis=0) / 2.0
    be := backends.Be()
    half := be.Float(0.5);
    return be.Mean(be.Square(y.Sub(t)), 0).Mul(half)
}

func(c *MeanSquared) Bprop(y backends.Value, t backends.Value) backends.Value {
    // (y - t) / y.shape[0]
    be := backends.Be()
    yShape0 := be.Float(float64(y.Shape()[0]))
    return y.Sub(t).Div(yShape0)
}

func(c *MeanSquared) ClassName() string {
    return "arhat.costs.MeanSquared"
}

//
//    SmoothL1Loss
//

type SmoothL1Loss struct {
    CostBase
    sigma float64
//    sigma2 backends.Value
}

var smoothL1LossInitArgMap = base.ArgMap{
    "sigma": base.NewFloatArgOpt(1.0),
}

func NewSmoothL1Loss(args ...interface{}) *SmoothL1Loss {
    c := new(SmoothL1Loss)
    c.Init(c, base.MakeArgs(args))
    return c
}

func(c *SmoothL1Loss) Init(self base.Object, args base.Args) {
    args = smoothL1LossInitArgMap.Expand(args)
    c.CostBase.Init(self)
    c.sigma = args["sigma"].(float64)
}

func(c *SmoothL1Loss) Call(y backends.Value, t backends.Value) backends.Value {
    // sum(smoothL1(y - t), axis=0)
    be := backends.Be()
    return be.Sum(c.smoothL1(y.Sub(t)), 0)
}

func(c *SmoothL1Loss) Bprop(y backends.Value, t backends.Value) backends.Value {
    // smoothL1grad(y - t)
    return c.smoothL1grad(y.Sub(t))
}

func(c *SmoothL1Loss) smoothL1(x backends.Value) backends.Value {
    // 0.5 * square(x) * sigma2 * (absolute(x) < 1 / sigma2) +
    //     (absolute(x) - 0.5 / sigma2) * (absolute(x) >= 1 / sigma2)
    be := backends.Be()
    half := be.Float(0.5)
    one := be.Float(1.0)
    sigma2 := be.Square(be.Float(c.sigma))
    return half.Mul(be.Square(x)).Mul(sigma2).Mul(be.Absolute(x).Lt(one.Div(sigma2))).Add(
        be.Absolute(x).Sub(half.Div(sigma2)).Mul(be.Absolute(x).Ge(one.Div(sigma2))))
}

func(c *SmoothL1Loss) smoothL1grad(x backends.Value) backends.Value {
    // x * sigma2 * (absolute(x) < 1 / sigma2) +
    //     sgn(x) * (absolute(x) >= 1 / sigma2)
    be := backends.Be()
    one := be.Float(1.0)
    sigma2 := be.Square(be.Float(c.sigma))
    return x.Mul(sigma2).Mul(be.Absolute(x).Lt(one.Div(sigma2))).Add(
        be.Sgn(x).Mul(be.Absolute(x).Ge(one.Div(sigma2))))
}

func(c *SmoothL1Loss) ClassName() string {
    return "arhat.costs.SmoothL1Loss"
}

//
//    SquareHingeLoss
//

type SquareHingeLoss struct {
    CostBase
    margin float64
}

var squareHingeLossInitArgMap = base.ArgMap{
    "margin": base.NewFloatArgOpt(1.0),
}

func NewSquareHingeLoss(args ...interface{}) *SquareHingeLoss {
    c := new(SquareHingeLoss)
    c.Init(c, base.MakeArgs(args))
    return c
}

func(c *SquareHingeLoss) Init(self base.Object, args base.Args) {
    args = squareHingeLossInitArgMap.Expand(args)
    c.CostBase.Init(self)
    c.margin = args["margin"].(float64)
}

func(c *SquareHingeLoss) Call(y backends.Value, t backends.Value) backends.Value {
    // t = 2 * t - 1
    // mean(square(maximum(margin - t * y, 0)), axis=0)
    be := backends.Be()
    zero := be.Float(0.0)
    one := be.Float(1.0)
    two := be.Float(2.0)
    margin := be.Float(c.margin)
    v := two.Mul(t).Sub(one)
    return be.Mean(be.Square(be.Maximum(margin.Sub(v.Mul(y)), zero)), 0)
}

func(c *SquareHingeLoss) Bprop(y backends.Value, t backends.Value) backends.Value {
    // t = 2 * t - 1
    // -2 * t * maximum(margin - t * y, 0) / y.shape[0]
    be := backends.Be()
    zero := be.Float(0.0)
    one := be.Float(1.0)
    two := be.Float(2.0)
    margin := be.Float(c.margin)
    yShape0 := be.Float(float64(y.Shape()[0]))
    v := two.Mul(t).Sub(one)
    return two.Neg().Mul(v).Mul(be.Maximum(margin.Sub(v.Mul(y)), zero)).Div(yShape0)
}

func(c *SquareHingeLoss) ClassName() string {
    return "arhat.costs.SquareHingeLoss"
}

//
//    Metric
//

type Metric interface {
    base.Object
    MetricNames() []string
    Call(y []backends.Tensor, t []backends.Tensor) []backends.Tensor
}

//
//    MetricBase
//

type MetricBase struct {
    base.ObjectBase
    metricNames []string
}

func(m *MetricBase) Init(self base.Object) {
    m.ObjectBase.Init(self, nil)
    m.metricNames = nil
}

func(m *MetricBase) MetricNames() []string {
    return m.metricNames
}

//
//    MultiMetric
//

type MultiMetric struct {
    MetricBase
    metric Metric
    index int
}

func NewMuitiMetric(metric Metric, index int) *MultiMetric {
    m := new(MultiMetric)
    m.Init(m, metric, index)
    return m
}

func(m *MultiMetric) Init(self base.Object, metric Metric, index int) {
    m.MetricBase.Init(self)
    m.metric = metric
    m.index = index
    m.metricNames = metric.MetricNames() // ACHTUNG: Not in original code
}

func(m *MultiMetric) Call(y []backends.Tensor, t []backends.Tensor) []backends.Tensor {
    // ACHTUNG: Apparent bug in original code: y used twice and t not used at all
    return m.metric.Call(y[m.index:m.index+1], t[m.index:m.index+1])
}

func(m *MultiMetric) ClassName() string {
    return "arhat.costs.MultiMetric"
}

// helpers for metric calculation

//
//    LogLoss
//

type LogLoss struct {
    MetricBase
    correctProbs backends.Tensor
}

func NewLogLoss() *LogLoss {
    m := new(LogLoss)
    m.Init(m)
    return m
}

func(m *LogLoss) Init(self base.Object) {
    m.MetricBase.Init(self)
    be := backends.Be()
    m.correctProbs = be.Iobuf([]int{1}, nil, base.DtypeNone, "", true, nil)
    m.metricNames = []string{"LogLoss"}
}

func(m *LogLoss) Call(y []backends.Tensor, t []backends.Tensor) []backends.Tensor {
    assertSingle(y)
    assertSingle(t)
    be := backends.Be()
    // correctProbs[] = sum(y * t, axis=0)
    // correctProbs[] = -safelog(correctProbs)
    m.correctProbs.Assign(be.Sum(y[0].Mul(t[0]), 0))
    m.correctProbs.Assign(be.Safelog(m.correctProbs).Neg())
    return []backends.Tensor{m.correctProbs}
}

func(m *LogLoss) ClassName() string {
    return "arhat.costs.LogLoss"
}

//
//    TopKMisclassification
//

type TopKMisclassification struct {
    MetricBase
    correctProbs backends.Tensor
    top1 backends.Tensor
    topk backends.Tensor
    k int
}

func NewTopKMisclassification(k int) *TopKMisclassification {
    m := new(TopKMisclassification)
    m.Init(m, k)
    return m
}

func(m *TopKMisclassification) Init(self base.Object, k int) {
    m.MetricBase.Init(self)
    be := backends.Be()
    m.correctProbs = be.Iobuf([]int{1}, nil, base.DtypeNone, "", true, nil)
    m.top1 = be.Iobuf([]int{1}, nil, base.DtypeNone, "", true, nil)
    m.topk = be.Iobuf([]int{1}, nil, base.DtypeNone, "", true, nil)
    m.k = k
    m.metricNames = []string{"LogLoss", "Top1Misclass", fmt.Sprintf("Top%dMisclass", k)}
}

func(m *TopKMisclassification) Call(y []backends.Tensor, t []backends.Tensor) []backends.Tensor {
    assertSingle(y)
    assertSingle(t)
    be := backends.Be()
    k := be.Int(m.k)
    izero := be.Int(0)
    ione := be.Int(1)
    fone := be.Float(1.0)
    // correctProbs[] = sum(y * t, axis=0)
    // nSlots = k - sum((y > correctProbs), axis=0)
    // nEq = sum((y == correctProbs), axis=0)
    // topk[] = 1.0 - (nslots > 0) * ((nEq <= nSlots) * (1 - nSlots / nEq) + nSlots / nEq)
    // top1[] = 1.0 - (max(y, axis=0) == self.correctProbs) / nEq
    // correctProbs[] = -safelog(correctProbs)
    m.correctProbs.Assign(be.Sum(y[0].Mul(t[0]), 0))
    nSlots := k.Sub(be.Sum(y[0].Gt(m.correctProbs), 0))
    nEq := be.Sum(y[0].Eq(m.correctProbs), 0)
    m.topk.Assign(fone.Sub(
        nSlots.Gt(izero).Mul(
            nEq.Ge(nSlots).Mul(ione.Sub(nSlots.Div(nEq))).Add(nSlots.Div(nEq)))))
    m.top1.Assign(fone.Sub(
        be.Max(y[0], 0).Eq(m.correctProbs).Div(nEq)))
    m.correctProbs.Assign(be.Safelog(m.correctProbs).Neg())
    return []backends.Tensor{m.correctProbs, m.top1, m.topk}
}

func(m *TopKMisclassification) ClassName() string {
    return "arhat.costs.TopKMisclassification"
}

//
//    Misclassification
//

type Misclassification struct {
    MetricBase
    preds backends.Tensor
    hyps backends.Tensor
    outputs backends.Tensor
}

func NewMisclassification(steps int) *Misclassification {
    m := new(Misclassification)
    m.Init(m, steps)
    return m
}

func(m *Misclassification) Init(self base.Object, steps int) {
    m.MetricBase.Init(self)
    be := backends.Be()
    m.preds = be.Iobuf([]int{1, steps}, nil, base.DtypeNone, "", false, nil)
    m.hyps = be.Iobuf([]int{1, steps}, nil, base.DtypeNone, "", false, nil)
    m.outputs = m.preds // Contains per record metric
    m.metricNames = []string{"Top1Misclass"}
}

func(m *Misclassification) Call(y []backends.Tensor, t []backends.Tensor) []backends.Tensor {
    assertSingle(y)
    assertSingle(t)
    be := backends.Be()
    // convert back from onehot and compare
    m.preds.Assign(be.Argmax(y[0], 0))
    m.hyps.Assign(be.Argmax(t[0], 0))
    m.outputs.Assign(be.NotEqual(m.preds, m.hyps))
    return []backends.Tensor{m.outputs}
}

func(m *Misclassification) ClassName() string {
    return "arhat.costs.Misclassification"
}

//
//    Accuracy
//

type Accuracy struct {
    MetricBase
    preds backends.Tensor
    hyps backends.Tensor
    outputs backends.Tensor
}

func NewAccuracy() *Accuracy {
    m := new(Accuracy)
    m.Init(m)
    return m
}

func(m *Accuracy) Init(self base.Object) {
    m.MetricBase.Init(self)
    be := backends.Be()
    m.preds = be.Iobuf([]int{1}, nil, base.DtypeNone, "", true, nil)
    m.hyps = be.Iobuf([]int{1}, nil, base.DtypeNone, "", true, nil)
    m.outputs = m.preds // Contains per record metric
    m.metricNames = []string{"Accuracy"}
}

func(m *Accuracy) Call(y []backends.Tensor, t []backends.Tensor) []backends.Tensor {
    be := backends.Be()
    // convert back from onehot and compare
    m.preds.Assign(be.Argmax(y[0], 0))
    m.hyps.Assign(be.Argmax(t[0], 0))
    m.outputs.Assign(be.Equal(m.preds, m.hyps))
    return []backends.Tensor{m.outputs}
}

func(m *Accuracy) ClassName() string {
    return "arhat.costs.Accuracy"
}

//
//    PrecisionRecall
//

type PrecisionRecall struct {
    MetricBase
    outputs [2]backends.Tensor
    tokenStats [3]backends.Tensor
    binBuf backends.Tensor
    eps float64
}

func NewPrecisionRecall(numClasses int, binarize bool, epsilon float64) *PrecisionRecall {
    m := new(PrecisionRecall)
    m.Init(m, numClasses, binarize, epsilon)
    return m
}

func(m *PrecisionRecall) Init(self base.Object, numClasses int, binarize bool, epsilon float64) {
    m.MetricBase.Init(self)
    be := backends.Be()
    m.outputs[0] = be.NewTensor([]int{numClasses}, base.DtypeNone)
    m.outputs[1] = be.NewTensor([]int{numClasses}, base.DtypeNone)
    m.tokenStats[0] = be.NewTensor([]int{numClasses}, base.DtypeNone)
    m.tokenStats[1] = be.NewTensor([]int{numClasses}, base.DtypeNone)
    m.tokenStats[2] = be.NewTensor([]int{numClasses}, base.DtypeNone)
    m.metricNames = []string{"Precision", "Recall"}
    if binarize {
        m.binBuf = be.Iobuf([]int{1}, nil, base.Int32, "", true, nil)
    } else {
        m.binBuf = nil
    }
    m.eps = epsilon
}

func(m *PrecisionRecall) Call(y []backends.Tensor, t []backends.Tensor) []backends.Tensor {
    assertSingle(y)
    assertSingle(t)
    be := backends.Be()
    eps := be.Float(m.eps)
    if m.binBuf != nil {
        // ACHTUNG: Originally was "out" version of argmax
        m.binBuf.Assign(be.Argmax(y[0], 0))
        y[0].Assign(be.Onehot(m.binBuf, 0))
    }
    // tokenStats[0][] = sum(y * t, axis=1)
    // tokenStats[1][] = sum(y, axis=1)
    // tokenStats[2][] = sum(t, axis=1)
    // outputs[0][] = tokenStats[0] / (tokenStats[1] + eps)
    // outputs[1][] = tokenStats[0] / (tokenStats[2] + eps)
    // True positives
    m.tokenStats[0].Assign(be.Sum(y[0].Mul(t[0]), 1))
    // Prediction
    m.tokenStats[1].Assign(be.Sum(y[0], 1))
    // Targets
    m.tokenStats[2].Assign(be.Sum(t[0], 1))
    // Precision
    m.outputs[0].Assign(m.tokenStats[0].Div(m.tokenStats[1].Add(eps)))
    // Recall
    m.outputs[1].Assign(m.tokenStats[0].Div(m.tokenStats[2].Add(eps)))
    return m.outputs[:]
}

func(m *PrecisionRecall) ClassName() string {
    return "arhat.costs.PrecisionRecall"
}

//
//    SKIPPED: ObjectDetection (Metric)
//    SKIPPED: BLEUScore (Metric)
//

//
//    SKIPPED: GANCost (Cost)
//

