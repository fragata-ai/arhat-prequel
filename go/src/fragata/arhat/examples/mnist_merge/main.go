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
Example demonstrating the use of merge layers training the MNIST data.

Usage:

    mkdir -p bin
    mkdir -p mnist_merge
    go build -o bin/mnist_merge fragata/arhat/examples/mnist_merge
    bin/mnist_merge -o mnist_merge
*/

import (
    "fragata/arhat/backends"
    "fragata/arhat/generators/callbacks"
    "fragata/arhat/generators/data"
    "fragata/arhat/generators/models"
    "fragata/arhat/generators/util"
    "fragata/arhat/initializers"
    "fragata/arhat/layers"
    "fragata/arhat/optimizers"
    "fragata/arhat/transforms"
)

var (
    Float = backends.NewFloat
    Callbacks = callbacks.NewCallbacks
    Mnist = data.NewMnist
    Gaussian = initializers.NewGaussian
    Affine = layers.NewAffine
    Sequential = layers.NewSequential
    MergeMultiStream = layers.NewMergeMultiStream
    GeneralizedCost = layers.NewGeneralizedCost
    Model = models.NewModel
    GradientDescentMomentum = optimizers.NewGradientDescentMomentum
    Rectlin = transforms.NewRectlin
    Logistic = transforms.NewLogistic
    CrossEntropyBinary = transforms.NewCrossEntropyBinary
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
    trainSet := dataset.TrainIter()
    validSet := dataset.ValidIter()

    // hyperparameters
    numEpochs := engine.Epochs()

    // weight initialization
    initNorm := Gaussian("loc", 0.0, "scale", 0.01)

    // initialize model
    path1 := Sequential(
        []layers.LayerItem{
            Affine("nout", 100, "init", initNorm, "activation", Rectlin()),
            Affine("nout", 100, "init", initNorm, "activation", Rectlin()),
        })

    path2 := Sequential(
        []layers.LayerItem{    
            Affine("nout", 100, "init", initNorm, "activation", Rectlin()),
            Affine("nout", 100, "init", initNorm, "activation", Rectlin()),
        })

    mlayers := []layers.LayerItem{
        MergeMultiStream([]layers.LayerItem{path1, path2}, "merge", "stack"),
        Affine("nout", 10, "init", initNorm, "activation", Logistic("shortcut", true)),
    }

    model := Model(mlayers)

    // setup cost fucntion
    cost := GeneralizedCost("costfunc", CrossEntropyBinary())

    // fit and validate
    optimizer := GradientDescentMomentum("learning_rate", 0.1, "momentum_coef", 0.9)

    // configure callbacks
    callbks := Callbacks()

    // run fit
    model.Fit(trainSet, cost, optimizer, numEpochs, callbks)

    errorRate := model.Eval(validSet, Misclassification(1))
    engine.Display("Misclassification error = %.1f%%", errorRate[0].Mul(Float(100.0)))
}

