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
// Ported from Python to Go and partly modified by FRAGATA COMPUTER SYSTEMS AG.
//

package optimizers

import (
    "fragata/arhat/backends"
    "fragata/arhat/base"
    "fragata/arhat/layers"
    "math"
)

//
//    Local helpers
//

func getParamList(layerList []layers.Layer) []layers.Param {
    var plist []layers.Param
    for _, l := range layerList {
        ptuple := l.GetParams()
        for _, p := range ptuple {
            plist = append(plist, p)
        }
    }
    return plist
}

//
//    Schedule
//

type Schedule interface {
    base.Object
    GetLearningRate(learningRate float64, epoch backends.Value) backends.Value
}

//
//    ScheduleArg
//

type ScheduleArg struct {
    base.ArgDefBase
}

func NewScheduleArg() *ScheduleArg {
    a := new(ScheduleArg)
    a.ArgDefBase.Init(true, nil)
    return a
}

func NewScheduleArgOpt(defval Schedule) *ScheduleArg {
    a := new(ScheduleArg)
    a.ArgDefBase.Init(false, defval)
    return a
}

func(a *ScheduleArg) Expand(v interface{}) (interface{}, bool) {
    if v == nil {
        return v, true
    }
    if _, ok := v.(Schedule); !ok {
        return nil, false
    }
    return v, true
}

func ToSchedule(v interface{}) Schedule {
    if v == nil {
        return nil
    }
    return v.(Schedule)
}

//
//    DefaultSchedule
//

// NOTE: All legacy functionality has been removed as adviced in the original code

type DefaultSchedule struct {
    base.ObjectBase
    stepConfig []int
    change float64
}

var defaultScheduleInitArgMap = base.ArgMap{
    "step_config": base.NewIntListArgOpt(nil),
    "change": base.NewFloatArgOpt(1.0),
}

func(s *DefaultSchedule) Init(self base.Object, args base.Args) {
    args = defaultScheduleInitArgMap.Expand(args)
    s.ObjectBase.Init(self, nil)
    s.stepConfig = base.ToIntList(args["step_config"])
    s.change = args["change"].(float64)
}

func(s *DefaultSchedule) StepConfig() []int {
    return s.stepConfig
}

func(s *DefaultSchedule) Change() float64 {
    return s.change
}

//
//    StepSchedule
//

type StepSchedule struct {
    base.ObjectBase
    stepConfig []int
    change []float64
}

var stepScheduleInitArgMap = base.ArgMap{
    "step_config": base.NewIntListArg(),
    "change": base.NewFloatListArg(),
}

func(s *StepSchedule) Init(self base.Object, args base.Args) {
    args = stepScheduleInitArgMap.Expand(args)
    s.ObjectBase.Init(self, nil)
    s.stepConfig = base.ToIntList(args["step_config"])
    s.change = base.ToFloatList(args["change"])
    base.AssertMsg(len(s.stepConfig) == len(s.change),
        "The arguments change and step_config must have the same length.")
}

func(s *StepSchedule) StepConfig() []int {
    return s.stepConfig
}

func(s *StepSchedule) Change() []float64 {
    return s.change
}

//
//    PowerSchedule
//

type PowerSchedule struct {
    base.ObjectBase
    stepConfig int
    change float64
}

var powerScheduleInitArgMap = base.ArgMap{
    "step_config": base.NewIntArg(),
    "change": base.NewFloatArg(),
}

func(s *PowerSchedule) Init(self base.Object, args base.Args) {
    args = powerScheduleInitArgMap.Expand(args)
    s.ObjectBase.Init(self, nil)
    s.stepConfig = args["step_config"].(int)
    s.change = args["change"].(float64)
}

func(s *PowerSchedule) StepConfig() int {
    return s.stepConfig
}

func(s PowerSchedule) Change() float64 {
    return s.change
}

//
//    Optimizer
//

// Need separate Reset method because in generator context states must be
// initialized outside optimization loop

type Optimizer interface {
    base.Object
    Reset(layerList []layers.Layer)
    Optimize(layerList []layers.Layer, epoch backends.Value)
    Schedule() Schedule
}

// 
//    OptimizerArg
//

type OptimizerArg struct {
    base.ArgDefBase
}

func NewOptimizerArg() *OptimizerArg {
    a := new(OptimizerArg)
    a.ArgDefBase.Init(true, nil)
    return a
}

func NewOptimizerArgOpt(defval Optimizer) *OptimizerArg {
    a := new(OptimizerArg)
    a.ArgDefBase.Init(false, defval)
    return a
}

func(a *OptimizerArg) Expand(v interface{}) (interface{}, bool) {
    if v == nil {
        return v, true
    }
    if _, ok := v.(Optimizer); !ok {
        return nil, false
    }
    return v, true
}

func ToOptimizer(v interface{}) Optimizer {
    if v == nil {
        return nil
    }
    return v.(Optimizer)
}

//
//    OptimizerBase
//

type OptimizerBase struct {
    base.ObjectBase
}

var optimizerBaseInitArgMap = base.ArgMap{
    "name": base.NewAnyArgOpt(""), // passthru
}

func(o *OptimizerBase) Init(self base.Object, args base.Args) {
    args = optimizerBaseInitArgMap.Expand(args)
    o.ObjectBase.Init(self, args.Filter([]string{"name"}))
}

func(o *OptimizerBase) Schedule() Schedule {
    return nil
}

func(o *OptimizerBase) ClipGradientNorm(
        paramList []layers.Param, clipNorm float64) backends.Value {
    be := backends.Be()
    if clipNorm == base.FloatNone {
        return be.Float(1.0)
    }
    var gradSquareSums backends.Value
    for i, _ := range paramList {
        sum := be.Sum(be.Square(paramList[i].Dw()), base.IntNone)
        if i == 0 {
            gradSquareSums = sum
        } else {
            gradSquareSums.Add(sum)
        }
    }
    gradNorm := be.NewTensor([]int{1, 1}, base.DtypeNone)
    gradNorm.Assign(be.Sqrt(gradSquareSums).Div(be.Int(be.Bsz())))
    g := gradNorm.GetScalar()
    c := be.Float(clipNorm)
    return c.Div(be.Maximum(g, c))
}

func(o *OptimizerBase) ClipValue(v backends.Value, absBound float64) backends.Value {
    if absBound != base.FloatNone {
        be := backends.Be()
        return be.Clip(v, be.Float(-math.Abs(absBound)), be.Float(math.Abs(absBound)))
    }
    return v
}

//
//    GradientDescentMomentum
//

type GradientDescentMomentum struct {
    OptimizerBase
    learningRate float64
    momentumCoef float64
    gradientClipNorm float64
    gradientClipValue float64
    paramClipValue float64
    wdecay float64
    schedule Schedule
    stochasticRound bool
    nesterov bool
}

var gradientDescentMomentumInitArgMap = base.ArgMap{
    "learning_rate": base.NewFloatArg(),
    "momentum_coef": base.NewFloatArg(),
    "stochastic_round": base.NewBoolArgOpt(false),
    "wdecay": base.NewFloatArgOpt(0.0),
    "gradient_clip_norm": base.NewFloatArgOpt(base.FloatNone),
    "gradient_clip_value": base.NewFloatArgOpt(base.FloatNone),
    "param_clip_value": base.NewFloatArgOpt(base.FloatNone),
    "name": base.NewAnyArgOpt(""), // passthru
    "schedule": NewScheduleArg(),
    "nesterov": base.NewBoolArgOpt(false),
}

func NewGradientDescentMomentum(args ...interface{}) *GradientDescentMomentum {
    o := new(GradientDescentMomentum)
    o.Init(o, base.MakeArgs(args))
    return o
}

func(o *GradientDescentMomentum) Init(self base.Object, args base.Args) {
    args = gradientDescentMomentumInitArgMap.Expand(args)
    o.OptimizerBase.Init(self, args.Filter([]string{"name"}))
    o.learningRate = args["learning_rate"].(float64)
    o.momentumCoef = args["momentum_coef"].(float64)
    o.gradientClipNorm = args["gradient_clip_norm"].(float64)
    o.gradientClipValue = args["gradient_clip_value"].(float64)
    o.paramClipValue = args["param_clip_value"].(float64)
    o.wdecay = args["wdecay"].(float64)
    o.schedule = ToSchedule(args["schedule"])
    o.stochasticRound = args["stochastic_round"].(bool)
    o.nesterov = args["nesterov"].(bool)
    if o.momentumCoef == 0.0 && o.nesterov {
        base.ValueError("nesterov requires non-zero momentum")
    }
}

func(o *GradientDescentMomentum) ClassName() string {
    return "arhat.optimizers.GradientDescentMomentum"
}

func(o *GradientDescentMomentum) Reset(layerList []layers.Layer) {
    if o.momentumCoef != 0.0 {
        be := backends.Be()
        paramList := getParamList(layerList)
        for i, _ := range paramList {
            dw := paramList[i].Dw()
            states := paramList[i].States()
// TODO: Enable (relevant for GPU backend)
//            w.SetRounding(o.stochasticRound)
//
            if len(states) == 0 {
                state := be.NewTensor(dw.Shape(), dw.Dtype())
                state.Fill(0)
                states = []backends.Tensor{state}
                paramList[i].SetStates(states)
            }
        }
    }
}

func(o *GradientDescentMomentum) Optimize(layerList []layers.Layer, epoch backends.Value) {
    be := backends.Be()

    lrate := o.schedule.GetLearningRate(o.learningRate, epoch)
    paramList := getParamList(layerList)

    scaleFactor := o.ClipGradientNorm(paramList, o.gradientClipNorm)

    for i, _ := range paramList {
        w := paramList[i].W()
        dw := paramList[i].Dw()
        states := paramList[i].States()
        if o.momentumCoef != 0.0 {
            base.Assert(len(states) != 0)
        }

        var param backends.Value = w
        var grad backends.Value = dw.Div(be.Int(be.Bsz()))
        grad = o.ClipValue(grad, o.gradientClipValue)

        if (o.momentumCoef == 0.0) {
            // ACHTUNG: Apparent bug in the original version: missing final assignment to param
            // param = (-lrate * scaleFactor) * grad + (1.0 - lrate * wdecay) * param
            one := be.Float(1.0)
            wdecay := be.Float(o.wdecay)
            param =
                lrate.Neg().Mul(scaleFactor).Mul(grad).Add(
                    one.Sub(lrate.Mul(wdecay)).Mul(param))
            w.Assign(o.ClipValue(param, o.paramClipValue))
        } else {
            // grad = scaleFactor * grad + wdecay * param
            // velocity[] = momentumCoef * velocity - lrate * grad
            wdecay := be.Float(o.wdecay)
            momentumCoef := be.Float(o.momentumCoef)
            grad = scaleFactor.Mul(grad).Add(wdecay.Mul(param))
            velocity := states[0]
            velocity.Assign(momentumCoef.Mul(velocity).Sub(lrate.Mul(grad)))

            // Nesterov accelerated gradient (NAG) is implemented the same
            // as in torch's "sgd.lua". It's a reformulation of Sutskever's
            // NAG equation found in "On the importance of initialization
            // and momentum in deep learning".
            if o.nesterov {
                // param += momentumCoef * velocity - lrate * grad
                param = 
                    param.Add(momentumCoef.Mul(velocity).Sub(lrate.Mul(grad)))
                w.Assign(o.ClipValue(param, o.paramClipValue))
            } else {
                // param += velocity
                w.Assign(o.ClipValue(param.Add(velocity), o.paramClipValue))
            }
        }
    }
}

func(o *GradientDescentMomentum) Schedule() Schedule {
    return o.schedule
}

//
//    MultiOptimizer
//

type MultiOptimizer struct {
    OptimizerBase
    optimizerMapping map[string]Optimizer
    mapList map[Optimizer][]layers.Layer
}

var multiOptimizerInitArgMap = base.ArgMap{
    "name": base.NewAnyArgOpt(""), // passthru
}

func NewMultiOptimizer(
        optimizerMapping map[string]Optimizer, args ...interface{}) *MultiOptimizer {
    o := new(MultiOptimizer)
    o.Init(o, optimizerMapping, base.MakeArgs(args))
    return o
}

func(o *MultiOptimizer) Init(
        self base.Object, optimizerMapping map[string]Optimizer, args base.Args) {
    args = multiOptimizerInitArgMap.Expand(args)
    o.OptimizerBase.Init(self, args.Filter([]string{"name"}))
    o.optimizerMapping = optimizerMapping
    base.AssertMsg(o.optimizerMapping["default"] != nil,
        "Must specify a default optimizer in layer type to optimizer mapping")
    o.mapList = nil
}

func(o *MultiOptimizer) ClassName() string {
    return "arhat.optimizers.MultiOptimizer"
}

func(o *MultiOptimizer) Reset(layerList []layers.Layer) {
    o.mapList = o.mapOptimizers(layerList)
    for opt, list := range o.mapList {
        opt.Reset(list)
    }
}

func(o *MultiOptimizer) Optimize(layerList []layers.Layer, epoch backends.Value) {
    for opt, list := range o.mapList {
        opt.Optimize(list, epoch)
    }
}

// maps the optimizers to their corresponding layers
func(o *MultiOptimizer) mapOptimizers(layerList []layers.Layer) map[Optimizer][]layers.Layer {
    mapList := make(map[Optimizer][]layers.Layer)
    for _, layer := range layerList {
        className := layer.ShortClassName()
        name := layer.Name()
        var opt Optimizer
        if v, ok := o.optimizerMapping[name]; ok {
            opt = v
        } else if v, ok := o.optimizerMapping[className]; ok {
            opt = v
        } else {
            opt = o.optimizerMapping["default"]
        }
        mapList[opt] = append(mapList[opt], layer)
    }
    return mapList
}

