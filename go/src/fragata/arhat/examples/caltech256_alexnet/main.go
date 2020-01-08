//
// Copyright 2019-2020 FRAGATA COMPUTER SYSTEMS AG
// Copyright 2015-2016 Intel Corporation
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
An Alexnet like model adapted to Arhat. Does not include the weight grouping.
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
    CannedImageData = data.NewCannedImageData
    Constant = initializers.NewConstant
    Gaussian = initializers.NewGaussian
    Affine = layers.NewAffine
    Conv = layers.NewConv
    Dropout = layers.NewDropout
    LRN = layers.NewLRN 
    Pooling = layers.NewPooling
    GeneralizedCost = layers.NewGeneralizedCost
    Model = models.NewModel
    GradientDescentMomentum = optimizers.NewGradientDescentMomentum
    MultiOptimizer = optimizers.NewMultiOptimizer
    PowerSchedule = optimizers.NewPowerSchedule
    Rectlin = transforms.NewRectlin
    Softmax = transforms.NewSoftmax
    CrossEntropyMulti = transforms.NewCrossEntropyMulti
    TopKMisclassification = transforms.NewTopKMisclassification
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
    dataset := CannedImageData(
        "path", engine.DataDir(), 
        "shape", []int{3, 224, 224},
        "nclass", 256,
        "pixel_mean", []int{104, 112, 127})
    train := dataset.TrainIter()
    valid := dataset.ValidIter()

    initG1 := Gaussian("scale", 0.01)
    initG2 := Gaussian("scale", 0.005)

    relu := Rectlin()

    mlayers := []layers.LayerItem{
        Conv(
            "fshape", []int{11, 11, 96},
            "padding", 0, 
            "strides", 4,
            "init", initG1, 
            "bias", Constant("val", 0.0), 
            "activation", relu, 
            "name", "conv1"),
        Pooling("fshape", 3, "strides", 2, "name", "pool1"),
        LRN("depth", 5, "ascale", 0.0001, "bpower", 0.75, "name", "norm1"),
        Conv(
            "fshape", []int{5, 5, 256}, 
            "padding", 2, 
            "init", initG1,
            "bias", Constant("val", 1.0), 
            "activation", relu, 
            "name", "conv2"),
        Pooling("fshape", 3, "strides", 2, "name", "pool2"),
        LRN("depth", 5, "ascale", 0.0001, "bpower", 0.75, "name", "norm2"),
        Conv(
            "fshape", []int{3, 3, 384}, 
            "padding", 1, 
            "init", initG1, 
            "bias", Constant("val", 0.0),
            "activation", relu, 
            "name", "conv3"),
        Conv(
            "fshape", []int{3, 3, 384}, 
            "padding", 1, 
            "init", initG1, 
            "bias", Constant("val", 1.0),
            "activation", relu, 
            "name", "conv4"),
        Conv(
            "fshape", []int{3, 3, 256}, 
            "padding", 1, 
            "init", initG1, 
            "bias", Constant("val", 1.0),
            "activation", relu, 
            "name", "conv5"),
        Pooling("fshape", 3, "strides", 2, "name", "pool5"),
        Affine(
            "nout", 4096, 
            "init", initG2, 
            "bias", Constant("val", 1.0),
            "activation", relu, 
            "name", "fc6"),
        Dropout("keep", 0.5, "name", "drop6"),
        Affine(
            "nout", 4096, 
            "init", initG2, 
            "bias", Constant("val", 1.0),
            "activation", relu, 
            "name", "fc7"),
        Dropout("keep", 0.5, "name", "drop7"),
        Affine(
            "nout", 256, 
            "init", initG1, 
            "bias", Constant("val", 0.0),
            "activation", Softmax(), 
            "name", "fc8"),
    }

    model := Model(mlayers)

    // scale LR by 0.1 every 20 epochs (this assumes batch_size = 256)
    weightSched := PowerSchedule("step_config", 20, "change", 0.1)
    optGdm := GradientDescentMomentum(
        "learning_rate", 0.01, 
        "momentum_coef", 0.9, 
        "wdecay", 0.0005, 
        "schedule", weightSched)
    optBiases := GradientDescentMomentum(
        "learning_rate", 0.02, 
        "momentum_coef", 0.9, 
        "schedule", weightSched)
    opt := MultiOptimizer(map[string]optimizers.Optimizer{"default": optGdm, "Bias": optBiases})

    // configure callbacks
    callbks := Callbacks()

    numEpochs := engine.Epochs()
    cost := GeneralizedCost("costfunc", CrossEntropyMulti())
    model.Fit(train, cost, opt, numEpochs, callbks)

    valmetric := TopKMisclassification(5)
    mets := model.Eval(valid, valmetric)

    engine.Display("Validation set metrics:")
    engine.Display("Logloss: %.2f, Accuracy: %.1f %% (Top-1), %.1f %% (Top-5)", 
        mets[0], 
        Float(1.0).Sub(mets[1]).Mul(Float(100.0)), 
        Float(1.0).Sub(mets[2]).Mul(Float(100.0)))
}

