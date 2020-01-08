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
    "fmt"
    "fragata/arhat/backends"
    "fragata/arhat/base"
    "fragata/arhat/generators"
    "fragata/arhat/optimizers"
)

//
//    Schedule
//

type Schedule interface {
    optimizers.Schedule
    generators.GlobalObject
}

//
//    ScheduleBase
//

type ScheduleBase struct {
    be generators.Generator
    index int
    symbol string
}

func(s *ScheduleBase) Init(self Schedule) {
    s.be = backends.Be().(generators.Generator)
    s.index = s.be.MakeIndex("schedule")
    s.symbol = fmt.Sprintf("schedule_%d", s.index)
    s.be.AddGlobalObject(self)
}

func(s *ScheduleBase) GetLearningRate(learningRate float64, epoch backends.Value) backends.Value {
    be := s.be
    epochSymbol := epoch.(*generators.IntSymbol).Symbol()
    result := be.MakeFloatSymbol()
    be.WriteLine("float %s = %s.GetLearningRate(%s, %s);", 
        result.Symbol(), s.symbol, generators.FormatFloat32(learningRate), epochSymbol)
    return result
}

//
//    DefaultSchedule
//

type DefaultSchedule struct {
    optimizers.DefaultSchedule
    ScheduleBase
}

func NewDefaultSchedule(args ...interface{}) *DefaultSchedule {
    s := new(DefaultSchedule)
    s.Init(s, base.MakeArgs(args))
    return s
}

func(s *DefaultSchedule) Init(self Schedule, args base.Args) {
    s.DefaultSchedule.Init(self, args)
    s.ScheduleBase.Init(self)
}

func(s *DefaultSchedule) ClassName() string {
    return "arhat.generators.optimizers.DefaultSchedule"
}

func(s *DefaultSchedule) Declare() {
    s.be.WriteLine("DefaultSchedule %s;", s.symbol)
}

func(s *DefaultSchedule) Initialize() {
    be := s.be
    be.WriteLine("%s.Init(%s);", s.symbol, generators.FormatFloat32(s.Change()))
    for _, step := range s.StepConfig() {
        be.WriteLine("%s.AddConfig(%d);", s.symbol, step)
    }
}

//
//    StepSchedule
//

type StepSchedule struct {
    optimizers.StepSchedule
    ScheduleBase
}

func NewStepSchedule(args ...interface{}) *StepSchedule {
    s := new(StepSchedule)
    s.Init(s, base.MakeArgs(args))
    return s
}

func(s *StepSchedule) Init(self Schedule, args base.Args) {
    s.StepSchedule.Init(self, args)
    s.ScheduleBase.Init(self)
}

func(s *StepSchedule) ClassName() string {
    return "arhat.generators.optimizers.StepSchedule"
}

func(s *StepSchedule) Declare() {
    s.be.WriteLine(fmt.Sprintf("StepSchedule %s;\n", s.symbol))
}

func(s *StepSchedule) Initialize() {
    be := s.be
    be.WriteLine(fmt.Sprintf("%s.Init();\n", s.symbol))
    step := s.StepConfig()
    change := s.Change()
    for i, _ := range step {
        be.WriteLine("%s.AddConfig(%d, %s);", 
            s.symbol, step[i], generators.FormatFloat32(change[i]))
    }
}

//
//    PowerSchedule
//

type PowerSchedule struct {
    optimizers.PowerSchedule
    ScheduleBase
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

func(s *PowerSchedule) Init(self Schedule, args base.Args) {
    s.PowerSchedule.Init(self, args)
    s.ScheduleBase.Init(self)
}

func(s *PowerSchedule) ClassName() string {
    return "arhat.generators.optimizers.PowerSchedule"
}

func(s *PowerSchedule) Declare() {
    s.be.WriteLine("PowerSchedule %s;", s.symbol)
}

func(s *PowerSchedule) Initialize() {
    s.be.WriteLine("%s.Init(%d, %s);", 
        s.symbol, s.StepConfig(), generators.FormatFloat32(s.Change()))
}

//
//    Optimizer
//

type Optimizer interface {
    optimizers.Optimizer
}

//
//    GradientDescentMomentum
//

type GradientDescentMomentum struct {
    optimizers.GradientDescentMomentum
}

func NewGradientDescentMomentum(args ...interface{}) *GradientDescentMomentum {
    o := new(GradientDescentMomentum)
    o.Init(o, base.MakeArgs(args))
    return o
}

func(o *GradientDescentMomentum) Init(self base.Object, args base.Args) {
    // is there any better solution?
    if _, ok := args["schedule"]; !ok {
        a := make(base.Args)
        for k, v := range args {
            a[k] = v
        }
        args = a
        args["schedule"] = NewDefaultSchedule()
    }
    o.GradientDescentMomentum.Init(self, args)
}

func(o *GradientDescentMomentum) ClassName() string {
    return "arhat.generators.optimizers.GradientDescentMomentum"
}

//
//    MultiOptimizer
//

type MultiOptimizer struct {
    optimizers.MultiOptimizer
}

func NewMultiOptimizer(
        optimizerMapping map[string]Optimizer, args ...interface{}) *MultiOptimizer {
    o := new(MultiOptimizer)
    o.Init(o, optimizerMapping, base.MakeArgs(args))
    return o
}

func(o *MultiOptimizer) Init(
        self base.Object, optimizerMapping map[string]Optimizer, args base.Args) {
    // is there any better solution?
    m := make(map[string]optimizers.Optimizer)
    for k, v := range optimizerMapping {
        m[k] = v
    }
    o.MultiOptimizer.Init(self, m, args)
}

func(o *MultiOptimizer) ClassName() string {
    return "arhat.generators.optimizers.MultiOptimizer"
}

