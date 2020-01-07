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
AllCNN style convnet on CIFAR10 data.

Reference:

    Striving for Simplicity: the All Convolutional Net `[Springenberg2015]`_
..  _[Springenberg2015]: http://arxiv.org/pdf/1412.6806.pdf

Usage:

    mkdir -p bin
    mkdir -p cifar10_allcnn
    go build -o bin/cifar10_allcnn fragata/arhat/examples/cifar10_allcnn
    bin/cifar10_allcnn -o cifar10_allcnn
*/

import (
    "fragata/arhat/backends"
    "fragata/arhat/generators/callbacks"
    "fragata/arhat/generators/data"
    "fragata/arhat/generators/models"
    "fragata/arhat/generators/optimizers"
    "fragata/arhat/generators/util"
    "fragata/arhat/initializers"
    "fragata/arhat/layers"
    "fragata/arhat/transforms"
)

var (
    Float = backends.NewFloat
    Callbacks = callbacks.NewCallbacks
    Cifar10 = data.NewCifar10
    Gaussian = initializers.NewGaussian
    Activation = layers.NewActivation
    Dropout = layers.NewDropout
    Pooling = layers.NewPooling
    Conv = layers.NewConv
    GeneralizedCost = layers.NewGeneralizedCost
    Model = models.NewModel
    GradientDescentMomentum = optimizers.NewGradientDescentMomentum
    Schedule = optimizers.NewDefaultSchedule
    Rectlin = transforms.NewRectlin
    Softmax = transforms.NewSoftmax
    CrossEntropyMulti = transforms.NewCrossEntropyMulti
    Misclassification = transforms.NewMisclassification
    Engine = util.NewEngine
    ArgParser = util.NewArgParser
)

func main() {
    // parse the command line arguments
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
        "normalize", false,
        "contrast_normalize", true,
        "whiten", true,
        "pad_classes", true)
    trainSet := dataset.TrainIter()
    validSet := dataset.ValidIter()

    // hyperparameters
    numEpochs := engine.Epochs()
    learningRate := 0.05 // make it configurable in the future
    weightDecay := 0.001 // -- " --

    // setup weight initialization function
    initNorm := Gaussian("scale", 0.05)

    relu := Rectlin()
    conv := []interface{}{
        "init", initNorm, 
        "batch_norm", false, 
        "activation", relu}
    convp1 := []interface{}{
        "init", initNorm, 
        "batch_norm", false, 
        "activation", relu, 
        "padding", 1}
    convp1s2 := []interface{}{
        "init", initNorm, 
        "batch_norm", false,
        "activation", relu, 
        "padding", 1, 
        "strides", 2}

    // setup model layers
    mlayers := []layers.LayerItem{
/* Temporarily disabled: CUDA Convolution Bprop does not support C=3
        Dropout("keep", 0.8),
*/
        Conv("fshape", []int{3, 3, 96}, "*", convp1),
        Conv("fshape", []int{3, 3, 96}, "*", convp1),
        Conv("fshape", []int{3, 3, 96}, "*", convp1s2),
        Dropout("keep", 0.5),
        Conv("fshape", []int{3, 3, 192}, "*", convp1),
        Conv("fshape", []int{3, 3, 192}, "*", convp1),
        Conv("fshape", []int{3, 3, 192}, "*", convp1s2),
        Dropout("keep", 0.5),
        Conv("fshape", []int{3, 3, 192}, "*", convp1),
        Conv("fshape", []int{1, 1, 192}, "*", conv),
        Conv("fshape", []int{1, 1, 16}, "*", conv),
        Pooling("fshape", 8, "op", "avg"),
        Activation("transform", Softmax()),
    }

    // initialize model object
    model := Model(mlayers)

    // setup cost function
    cost := GeneralizedCost("costfunc", CrossEntropyMulti())

    // setup optimizer
    optGdm := GradientDescentMomentum(
        "learning_rate", learningRate, 
        "momentum_coef", 0.9,
        "wdecay", weightDecay,
        "schedule", Schedule("step_config", []int{200, 250, 300}, "change", 0.1))

    // configure callbacks
    callbks := Callbacks()

    model.Fit(trainSet, cost, optGdm, numEpochs, callbks)

    errorRate := model.Eval(validSet, Misclassification(1))
    engine.Display("Misclassification error = %.1f%%", errorRate[0].Mul(Float(100.0)))
}

