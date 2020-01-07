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
// Ported from Python to Go and partly modified by FRAGATA COMPUTER SYSTEMS AG.
//

package util

import (
    "fmt"
    "fragata/arhat/backends"
    "fragata/arhat/base"
    "fragata/arhat/generators"
    "fragata/arhat/generators/cpu"
    "fragata/arhat/generators/cuda"
)

//
//    Engine
//

type MainFunc func(engine *Engine)

type Engine struct {
    base.ObjectBase
    backend string
    output string
    dataDir string
    batchSize int
    epochs int
    be generators.Generator
}

var engineInitArgMap = base.ArgMap{
    "name": base.NewAnyArgOpt(""), // passthru
    "backend": base.NewStringArgOpt("cuda"),
    "output": base.NewStringArgOpt("./output"),
    "data_dir": base.NewStringArgOpt("./data"),
    "batch_size": base.NewIntArgOpt(128),
    "epochs": base.NewIntArgOpt(10),
}

func NewEngine(args ...interface{}) *Engine {
    e := new(Engine)
    e.Init(e, base.MakeArgs(args))
    return e
}

func(e *Engine) Init(self base.Object, args base.Args) {
    args = engineInitArgMap.Expand(args)
    e.ObjectBase.Init(self, args.Filter([]string{"name"}))
    e.backend = args["backend"].(string)
    e.output = args["output"].(string)
    e.dataDir = args["data_dir"].(string)
    e.batchSize = args["batch_size"].(int)
    e.epochs = args["epochs"].(int)
    e.initBackend()
}

func(e *Engine) Backend() string {
    return e.backend
}

func(e *Engine) Output() string {
    return e.output
}

func(e *Engine) DataDir() string {
    return e.dataDir
}

func(e *Engine) BatchSize() int {
    return e.batchSize
}

func(e *Engine) Epochs() int {
    return e.epochs
}

func(e *Engine) Be() generators.Generator {
    return e.be
}

func(e *Engine) initBackend() {
    switch e.backend {
    case "cpu":
        e.initCpuBackend()
    case "cuda":
        e.initCudaBackend()
    default:
        base.ValueError("Invalid backend type: %s", e.backend)
    }
}

func(e *Engine) initCpuBackend() {
    be := 
        cpu.NewCpuGenerator(
            base.IntNone,            // rngSeed
            base.DtypeNone,          // defaultDtype
            base.IntNone,            // stochasticRound
            false,                   // bench
            base.IntNone,            // scratchSize
            base.IntNone,            // histBins
            base.IntNone,            // histOffset
            backends.CompatModeNone) // compatMode
    be.SetBsz(e.batchSize)
    backends.SetBe(be)
    e.be = be
}

func(e *Engine) initCudaBackend() {
    be := 
        cuda.NewCudaGenerator(
            base.IntNone,            // rngSeed
            base.DtypeNone,          // defaultDtype
            base.IntNone,            // stochasticRound
            base.IntNone,            // deviceId
            [2]int{6, 0},            // computeCapability
            false,                   // bench
            base.IntNone,            // scratchSize
            base.IntNone,            // histBins
            base.IntNone,            // histOffset
            backends.CompatModeNone) // compatMode
    be.SetBsz(e.batchSize)
    backends.SetBe(be)
    e.be = be
}

func(e *Engine) ClassName() string {
    return "fragata.arhat.generators.util.Engine"
}

func(e *Engine) Run(mainFunc MainFunc) {
    be := e.be

    // start main function
    be.StartCode()
    be.WriteLine("void Main() {")
    be.Indent(1)

    be.WriteLine("Prolog();")

    mainFunc(e)

    // end main function
    be.Indent(-1)
    be.WriteLine("}")
    be.WriteLine("")

    be.EndCode()

    // output generated code
    err := be.OutputCode(e.output)
    if err != nil {
        base.RuntimeError("%s", err.Error())
    }
}

func(e *Engine) Display(format string, args ...interface{}) {
    line := fmt.Sprintf(`printf("%s\n"`, format)
    for _, arg := range args {
        line += ", "
        switch v := arg.(type) {
        case backends.Value:
            line += generators.FormatScalar(v)
        case int:
            line += fmt.Sprintf("%d", v)
        case float64:
            line += generators.FormatFloat32(v)
        case string:
            line += `"` + v + `"`
        }
    }
    line += ");"
    e.be.WriteLine("%s", line)
}

