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
Small MLP with fully connected layers trained on CIFAR10 data.

Usage:

    mkdir -p bin
    mkdir -p cifar10
    go build -o bin/cifar10 fragata/arhat/examples/cifar10
    bin/cifar10 -o cifar10
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
    Cifar10 = data.NewCifar10
    Uniform = initializers.NewUniform
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
    // load up the CIFAR10 data set
    dataset := Cifar10(
        "path", engine.DataDir(),
        "normalize", true,
        "contrast_normalize", false,
        "whiten", false)
    train := dataset.TrainIter()
    test := dataset.ValidIter()

    // setup weight initialization function
    initUni := Uniform("low", -0.1, "high", 0.1)

    // setup model layers
    mlayers := []layers.LayerItem{
        Affine(
            "nout", 200, 
            "init", initUni, 
            "activation", Rectlin()),
        Affine(
            "nout", 10, 
            "init", initUni, 
            "activation", Logistic("shortcut", true)),
    }

    // initialize model object
    model := Model(mlayers)

    // setup cost function
    cost := GeneralizedCost("costfunc", CrossEntropyBinary())

    // setup optimizer
    optGdm := GradientDescentMomentum(
        "learning_rate", 0.01, 
        "momentum_coef", 0.9)

    // configure callbacks
    callbks := Callbacks()

    model.Fit(train, cost, optGdm, engine.Epochs(), callbks)

    errorRate := model.Eval(test, Misclassification(1))
    engine.Display("Misclassification error = %.1f%%", errorRate[0].Mul(Float(100.0)))
}

