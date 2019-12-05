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
Trains a small multi-layer perceptron with fully connected layers on MNIST data.

Usage (CUDA backend, default):

    mkdir -p bin
    mkdir -p mnist_mlp_cuda
    go build -o bin/mnist_mlp fragata/arhat/examples/mnist_mlp
    bin/mnist_mlp -o mnist_mlp_cuda

Usage (CPU backend):

    mkdir -p bin
    mkdir -p mnist_mlp_cpu
    go build -o bin/mnist_mlp fragata/arhat/examples/mnist_mlp
    bin/mnist_mlp -b cpu -o mnist_mlp_cpu
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

    // setup cost fucntion
    cost := GeneralizedCost("costfunc", CrossEntropyBinary())

    // setup optimizer
    optimizer := GradientDescentMomentum( 
        "learning_rate", 0.1, 
        "momentum_coef", 0.9, 
        "stochastic_round", false)

    // configure callbacks
    callbks := Callbacks()

    // run fit
    mlp.Fit(trainSet, cost, optimizer, engine.Epochs(), callbks)

    errorRate := mlp.Eval(validSet, Misclassification(1))
    engine.Display("Misclassification error = %.1f%%", errorRate[0].Mul(Float(100.0)))
}

