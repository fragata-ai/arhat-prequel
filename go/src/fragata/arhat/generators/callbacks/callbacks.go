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

package callbacks

import (
    "fmt"
    "fragata/arhat/backends"
    "fragata/arhat/base"
    "fragata/arhat/generators"
)

//
//    NOTE: This is a temporary skeleton version; to be expanded later
//

//
//    Callbacks
//

type Callbacks struct {
    base.ObjectBase
    be generators.Generator
    index int
    symbol string
}

var callbacksInitArgMap = base.ArgMap{
    "name": base.NewAnyArgOpt(""), // passthru
}

func NewCallbacks(args ...interface{}) *Callbacks {
    c := new(Callbacks)
    c.Init(c, base.MakeArgs(args))
    return c
}

func(c *Callbacks) Init(self base.Object, args base.Args) {
    args = callbacksInitArgMap.Expand(args)
    c.ObjectBase.Init(self, args.Filter([]string{"name"}))
    c.be = backends.Be().(generators.Generator)
    c.index = c.be.MakeIndex("callbacks")
    c.symbol = fmt.Sprintf("callbacks_%d", c.index)
    c.be.AddGlobalObject(c) // assume no derived classes
}

func(c *Callbacks) ClassName() string {
    return "fragata.arhat.generators.callbacks.Callbacks"
}

func(c *Callbacks) Index() int {
    return c.index
}

func(c *Callbacks) Symbol() string {
    return c.symbol
}

func(c *Callbacks) Declare() {
    c.be.WriteLine("Callbacks %s;", c.symbol)
}

func(c *Callbacks) Initialize() {
    // nothing to do so far
}

