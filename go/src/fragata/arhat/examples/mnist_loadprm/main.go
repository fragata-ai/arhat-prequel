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

package main

/*
Loads network parameters from a file previously saved by mnist_saveprm,
evaluates the network, and performs inference storing results in memory
using the array data writer.

Usage:

    mkdir -p bin
    mkdir -p mnist_loadprm
    go build -o bin/mnist_loadprm fragata/arhat/examples/mnist_loadprm
    bin/mnist_loadprm -o mnist_loadprm
*/

import (
    "fmt"
    "fragata/arhat/backends"
    "fragata/arhat/base"
    "fragata/arhat/generators/data"
    "fragata/arhat/generators/models"
    "fragata/arhat/generators/util"
    "fragata/arhat/initializers"
    "fragata/arhat/layers"
    "fragata/arhat/transforms"
)

var (
    Float = backends.NewFloat
    Mnist = data.NewMnist
    ArrayWriter = data.NewArrayWriter
    Gaussian = initializers.NewGaussian
    Affine = layers.NewAffine
    Model = models.NewModel
    Rectlin = transforms.NewRectlin
    Logistic = transforms.NewLogistic
    Misclassification = transforms.NewMisclassification
    Engine = util.NewEngine
    ArgParser = util.NewArgParser
)

func main() {
    parser := ArgParser()
    parser.Parse()
    args := parser.Args()

    engine := Engine(args...)
    engine.Run(Main)
}

func Main(engine *util.Engine) {
    // load up the MNIST data set
    dataset := Mnist("path", engine.DataDir())
    validSet := dataset.ValidIter()

    // setup weight initialization function
    initNorm := Gaussian("loc", 0.0, "scale", 0.01)

    // setup model layers
    mlayers := []layers.LayerItem{
        Affine(
            "nout", 100, 
            "init", initNorm, 
            "activation", Rectlin()),
        Affine(
            "nout", 10, 
            "init", initNorm, 
            "activation", Logistic("shortcut", true)),
    }

    // initialize model object
    mlp := Model(mlayers)

    // initialize model (required before loading parameters)
    mlp.Initialize(validSet.Shape(), nil)

    // load model parameters
    mlp.LoadParams("params/mnist_mlp.prm")

    errorRate := mlp.Eval(validSet, Misclassification(1))
    engine.Display("Misclassification error = %.1f%%", errorRate[0].Mul(Float(100.0)))

    writer := ArrayWriter([]int{10}, base.Float32, "")
    mlp.GetOutputs(validSet, writer)

    displayOutputs(engine, writer)
}

var displayOutputsCode = `
    printf("Output\n");
    ArrayWriter *w = &%s;
    int ndata = w->Len();
    for (int i = 0; i < 10; i++) {
        float *p = (float *)w->Buffer(i);
        printf("[%%d] %%.2f %%.2f %%.2f %%.2f %%.2f %%.2f %%.2f %%.2f %%.2f %%.2f\n",
            i, p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8], p[9]);
    }
    printf("...\n");
    for (int i = 0; i < 10; i++) {
        int k = ndata - 10 + i;
        float *p = (float *)w->Buffer(k);
        printf("[%%d] %%.2f %%.2f %%.2f %%.2f %%.2f %%.2f %%.2f %%.2f %%.2f %%.2f\n",
            k, p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8], p[9]);
    }
`

func displayOutputs(engine *util.Engine, writer *data.ArrayWriter) {
    be := engine.Be()
    chunk := fmt.Sprintf(displayOutputsCode, writer.Symbol())
    be.WriteChunk(chunk)
}

