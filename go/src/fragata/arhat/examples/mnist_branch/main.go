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

package main

/*
Example that trains a small multi-layer perceptron with multiple branches on MNIST data.

Branch nodes are used to indicate points at which different layer sequences diverge

The topology of the network is:

 cost1      cost3
  |          /
 m_l4      b2_l2
  |        /
  | ___b2_l1
  |/
 m_l3       cost2
  |          /
 m_l2      b1_l2
  |        /
  | ___b1_l1
  |/
  |
 m_l1
  |
  |
 data

Usage:

    mkdir -p bin
    mkdir -p mnist_branch
    go build -o bin/mnist_branch fragata/arhat/examples/mnist_branch
    bin/mnist_branch -o mnist_branch
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
    Mnist = data.NewMnist
    Gaussian = initializers.NewGaussian
    Affine = layers.NewAffine
    BranchNode = layers.NewBranchNode
    Sequential = layers.NewSequential
    SingleOutputTree = layers.NewSingleOutputTree
    GeneralizedCost = layers.NewGeneralizedCost
    Multicost = layers.NewMulticost
    Model = models.NewModel
    GradientDescentMomentum = optimizers.NewGradientDescentMomentum
    Rectlin = transforms.NewRectlin
    Logistic = transforms.NewLogistic
    Softmax = transforms.NewSoftmax
    CrossEntropyBinary = transforms.NewCrossEntropyBinary
    CrossEntropyMulti = transforms.NewCrossEntropyMulti
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

    normrelu := []interface{}{"init", initNorm, "activation", Rectlin()}
    normsigm := []interface{}{"init", initNorm, "activation", Logistic("shortcut", true)}
    normsoft := []interface{}{"init", initNorm, "activation", Softmax()}

    // setup model layers
    b1 := BranchNode("name", "b1")
    b2 := BranchNode("name", "b2")

    p1 := Sequential(
        []layers.LayerItem{
            Affine("nout", 100, "name", "m_l1", "*", normrelu),
            b1,
            Affine("nout", 32, "name", "m_l2", "*", normrelu),
            Affine("nout", 16, "name", "m_l3", "*", normrelu),
            b2,
            Affine("nout", 10, "name", "m_l4", "*", normsoft),
        })

    p2 := Sequential(
        []layers.LayerItem{
            b1,
            Affine("nout", 16, "name", "b1_l1", "*", normrelu),
            Affine("nout", 10, "name", "b1_l2", "*", normsigm),
        })

    p3 := Sequential(
        []layers.LayerItem{
            b2,
            Affine("nout", 16, "name", "b2_l1", "*", normrelu),
            Affine("nout", 10, "name", "b2_l2", "*", normsigm),
        })

    // initialize model object
    alphas := []float64{1.0, 0.25, 0.25}
    mlp := Model(
        SingleOutputTree([]layers.LayerItem{p1, p2, p3}, 
        "alphas", alphas))

    // setup cost function as CrossEntropy
    cost := Multicost(
        []layers.Cost{
            GeneralizedCost("costfunc", CrossEntropyMulti()),
            GeneralizedCost("costfunc", CrossEntropyBinary()),
            GeneralizedCost("costfunc", CrossEntropyBinary()),
        },
        "weights", []float64{1.0, 0.0, 0.0})

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

