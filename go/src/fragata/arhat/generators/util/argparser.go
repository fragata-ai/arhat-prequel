//
// Copyright 2019-2020 FRAGATA COMPUTER SYSTEMS AG
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

import "flag"

//
//    ArgParser
//

const (
    defaultBackend = "cuda"
    defaultOutput = "./output"
    defaultDataDir = "./data"
    defaultBatchSize = 128
    defaultEpochs = 10
)

const (
    usageBackend = "backend type [cpu|cuda]"
    usageOutput = "output directory for generated code"
    usageDataDir = "working directory in which to cache downloaded and preprocessed datasets"
    usageBatchSize = "batch size"
    usageEpochs = "number of complete passes over the dataset to run"
)

type ArgParser struct {
    backend string
    output string
    dataDir string
    batchSize int
    epochs int
}

func NewArgParser() *ArgParser {
    p := new(ArgParser)
    p.Init()
    return p
}

func(p *ArgParser) Init() {
    p.backend = defaultBackend
    p.output = defaultOutput
    p.dataDir = defaultDataDir
    p.batchSize = defaultBatchSize
    p.epochs = defaultEpochs
}

func(p *ArgParser) Parse() {
    stringVar(&p.backend, []string{"b", "backend"}, defaultBackend, usageBackend)
    stringVar(&p.output, []string{"o", "output"}, defaultOutput, usageOutput)
    stringVar(&p.dataDir, []string{"w", "data_dir"}, defaultDataDir, usageDataDir)
    intVar(&p.batchSize, []string{"z", "batch_size"}, defaultBatchSize, usageBatchSize)
    intVar(&p.epochs, []string{"e", "epochs"}, defaultEpochs, usageEpochs)
    flag.Parse()
}

func(p *ArgParser) Args() []interface{} {
    // args in format suitable for Engine construction
    return []interface{}{
        "backend", p.backend,
        "output", p.output,
        "data_dir", p.dataDir,
        "batch_size", p.batchSize,
        "epochs", p.epochs,
    }
}

func(p *ArgParser) Backend() string {
    return p.backend
}

func(p *ArgParser) Output() string {
    return p.output
}

func(p *ArgParser) DataDir() string {
    return p.dataDir
}

func(p *ArgParser) BatchSize() int {
    return p.batchSize
}

func(p *ArgParser) Epochs() int {
    return p.epochs
}

func intVar(p *int, names []string, value int, usage string) {
    for _, name := range names {
        flag.IntVar(p, name, value, usage)
    }
}

func stringVar(p *string, names []string, value string, usage string) {
    for _, name := range names {
        flag.StringVar(p, name, value, usage)
    }
}

