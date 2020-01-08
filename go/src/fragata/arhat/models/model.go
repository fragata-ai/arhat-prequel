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

package models

import (
    "fragata/arhat/base"
    "fragata/arhat/layers"
    "fragata/arhat/optimizers"
)

//
//    Model
//

type Model struct {
    base.ObjectBase
    layers layers.LayerContainerTrait
    optimizer optimizers.Optimizer
    initialized bool
    cost layers.Cost
    globalDeltas *layers.DeltasTree
}

var modelInitArgMap = base.ArgMap{
    "name": base.NewAnyArgOpt(""), // passthru
    "weights_only": base.NewBoolArgOpt(false),
    "optimizer": optimizers.NewOptimizerArgOpt(nil),
}

func NewModel(argLayers interface{}, args ...interface{}) *Model {
    m := new(Model)
    m.Init(m, argLayers, base.MakeArgs(args))
    return m
}

func(m *Model) Init(self base.Object, argLayers interface{}, args base.Args) {
    args = modelInitArgMap.Expand(args)
    m.ObjectBase.Init(self, args.Filter([]string{"name"}))
    m.optimizer = optimizers.ToOptimizer(args["optimizer"])
    m.initialized = false
    m.cost = nil
    switch v := argLayers.(type) {
    case layers.LayerContainerTrait:
        m.layers = v
        // SKIPPED: Special support for SkipThought
    case []layers.LayerItem:
        //  Wrap the list of layers in a Sequential container
        m.layers = layers.NewSequential(v)
    default:
        base.TypeError("Invalid argument: layers")
    }
}

func(m *Model) ClassName() string {
    return "arhat.models.Model"
}

func(m *Model) Layers() layers.LayerContainerTrait {
    return m.layers
}

func(m *Model) Optimizer() optimizers.Optimizer {
    return m.optimizer
}

func(m *Model) Initialized() bool {
    return m.initialized
}

func(m *Model) Cost() layers.Cost {
    return m.cost
}

func(m *Model) LayersToOptimize() []layers.Layer {
    return m.layers.LayersToOptimize()
}

func(m *Model) Initialize(inObj []int, cost layers.Cost) {
    if m.initialized {
        return
    }

    // Propagate shapes through the layers to configure
    m.layers.Configure(layers.MakeInputObject(inObj))

    if cost != nil {
        cost.Initialize(m.layers)
        m.cost = cost
    }

    // Now allocate space
    m.layers.Allocate(nil, false)
    m.layers.AllocateDeltas(nil)
    m.initialized = true
}

func(m *Model) AllocateDeltas() {
    if m.globalDeltas == nil {
        m.globalDeltas = layers.NewDeltasTree(nil)
        m.layers.AllocateDeltas(m.globalDeltas)
        // allocate the buffers now that all the
        // nesting and max sizes have been determined
        m.globalDeltas.AllocateBuffers()
    }
    // set the deltas
    m.layers.SetDeltas(m.globalDeltas)
}

func(m *Model) String() string {
    return "Network Layers:\n" + m.layers.NestedStr(0)
}

func(m *Model) ReadParams(r layers.ParamReader) {
    m.layers.ReadParams(r)
}

func(m *Model) WriteParams(w layers.ParamWriter) {
    m.layers.WriteParams(w)
}

