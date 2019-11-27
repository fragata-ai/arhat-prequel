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

package optimizers

import (
    "fmt"
    "fragata/arhat/backends"
    "fragata/arhat/base"
    "fragata/arhat/layers"
    "math"
    "strings"
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

// ACHTUNG: Same as in generators - unify?
func formatFloat32(x float64) string {
    s := fmt.Sprintf("%.7g", x)
    if !strings.Contains(s, ".") && !strings.Contains(s, "e") {
        s += ".0"
    }
    s += "f"
    return s
}

//
//    Schedule
//

type Schedule interface {
    base.Object
    GetLearningRate(learningRate float64, epoch int) float64
    Construct(symbol string) []string
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
//    ScheduleBase
//

type ScheduleBase struct {
    base.ObjectBase
}

func(s *ScheduleBase) Init(self base.Object) {
    s.ObjectBase.Init(self, nil)
}

//
//    DefaultSchedule
//

// NOTE: All legacy functionality has been removed as adviced in the original code

type DefaultSchedule struct {
    ScheduleBase
    stepConfig []int
    change float64
    steps int
}

var defaultScheduleInitArgMap = base.ArgMap{
    "step_config": base.NewIntListArgOpt(nil),
    "change": base.NewFloatArgOpt(1.0),
}

func NewDefaultSchedule(args ...interface{}) *DefaultSchedule {
    s := new(DefaultSchedule)
    s.Init(s, base.MakeArgs(args))
    return s
}

func(s *DefaultSchedule) Init(self base.Object, args base.Args) {
    args = defaultScheduleInitArgMap.Expand(args)
    s.ScheduleBase.Init(self)
    s.stepConfig = base.ToIntList(args["step_config"])
    s.change = args["change"].(float64)
    s.steps = 0
}

func(s *DefaultSchedule) GetLearningRate(learningRate float64, epoch int) float64 {
    // NOTE: All legacy functionality has been removed to StepSchedule and PowerSchedule
    //     as adviced in the original code
    steps := 0
    for _, v := range s.stepConfig {
        if epoch >= v {
            steps++
        }
    }
    s.steps = steps
    return learningRate * math.Pow(s.change, float64(s.steps))
}

func(s *DefaultSchedule) Construct(symbol string) []string {
    var result []string
    result = append(result, 
        fmt.Sprintf("DefaultSchedule %s(%s);", symbol, formatFloat32(s.change)))
    for _, step := range s.stepConfig {
        result = append(result, fmt.Sprintf("%s.AddConfig(%d);", symbol, step))
    }
    return result
}

func(s *DefaultSchedule) ClassName() string {
    return "arhat.optimizers.DefaultSchedule"
}

//
//    StepSchedule
//

type StepSchedule struct {
    ScheduleBase
    stepConfig []int
    change []float64
    steps float64
}

var stepScheduleInitArgMap = base.ArgMap{
    "step_config": base.NewIntListArg(),
    "change": base.NewFloatListArg(),
}

func NewStepSchedule(args ...interface{}) *StepSchedule {
    s := new(StepSchedule)
    s.Init(s, base.MakeArgs(args))
    return s
}

func(s *StepSchedule) Init(self base.Object, args base.Args) {
    args = stepScheduleInitArgMap.Expand(args)
    s.ScheduleBase.Init(self)
    s.stepConfig = base.ToIntList(args["step_config"])
    s.change = base.ToFloatList(args["change"])
    s.steps = 0.0
    base.AssertMsg(len(s.stepConfig) == len(s.change),
        "The arguments change and step_config must have the same length.")
}

func(s *StepSchedule) GetLearningRate(learningRate float64, epoch int) float64 {
    for i, v := range s.stepConfig {
        if epoch == v {
            s.steps = s.change[i]
            break
        }
    }
    if s.steps != 0.0 {
        return s.steps
    }
    return learningRate
}

func(s *StepSchedule) Construct(symbol string) []string {
    var result []string
    result = append(result, fmt.Sprintf("StepSchedule %s;\n", symbol))
    for i, _ := range s.stepConfig {
        result = append(result, 
            fmt.Sprintf("%s.AddConfig(%d, %s);", 
                symbol, s.stepConfig[i], formatFloat32(s.change[i])))
    }
    return result
}

func(s *StepSchedule) ClassName() string {
    return "arhat.optimizers.StepSchedule"
}

//
//    PowerSchedule
//

type PowerSchedule struct {
    ScheduleBase
    stepConfig int
    change float64
    steps int
}

var powerScheduleInitArgMap = base.ArgMap{
    "step_config": base.NewIntArg(),
    "change": base.NewFloatArg(),
}

func NewPowerSchedule(args ...interface{}) *PowerSchedule {
    s := new(PowerSchedule)
    s.Init(s, base.MakeArgs(args))
    return s
}

func(s *PowerSchedule) Init(self base.Object, args base.Args) {
    args = powerScheduleInitArgMap.Expand(args)
    s.ScheduleBase.Init(self)
    s.stepConfig = args["step_config"].(int)
    s.change = args["change"].(float64)
    s.steps = 0
}

func(s *PowerSchedule) GetLearningRate(learningRate float64, epoch int) float64 {
    s.steps = epoch / s.stepConfig
    return learningRate * math.Pow(s.change, float64(s.steps))
}

func(s *PowerSchedule) Construct(symbol string) []string {
    var result []string
    result = append(result, 
        fmt.Sprintf("PowerSchedule %s(%d, %s);", 
            symbol, s.stepConfig, formatFloat32(s.change)))
    return result
}

func(s *PowerSchedule) ClassName() string {
    return "arhat.optimizers.PowerSchedule"
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
    "schedule": NewScheduleArgOpt(nil),
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
    if o.schedule == nil {
        o.schedule = NewDefaultSchedule()
    }
    o.stochasticRound = args["stochastic_round"].(bool)
    o.nesterov = args["nesterov"].(bool)
    if o.momentumCoef == 0.0 && o.nesterov {
        base.ValueError("nesterov requires non-zero momentum")
    }
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

    lrate := be.GetLearningRate(o.schedule, o.learningRate, epoch)
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

func(o *GradientDescentMomentum) ClassName() string {
    return "arhat.optimizers.GradientDescentMomentum"
}

