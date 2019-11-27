//
// Copyright 2019 FRAGATA COMPUTER SYSTEMS AG
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

package data

import (
    "fragata/arhat/backends"
    "fragata/arhat/base"
    "fragata/arhat/generators"
)

//
//    DataSet
//

type DataSet interface {
    base.Object
    TrainIter() *DataIterator
    ValidIter() *DataIterator
    TestIter() *DataIterator
    Declare()
    Initialize()
}

//
//    DataSetBase
//

type DataSetBase struct {
    base.ObjectBase
    be generators.Generator
    iterMap map[string]*DataIterator
    index int
}

var dataSetBaseInitArgMap = base.ArgMap{
    "name": base.NewAnyArgOpt(""), // passthru
}

func(d *DataSetBase) Init(self DataSet, args base.Args) {
    d.ObjectBase.Init(self, args.Filter([]string{"name"}))
    args = dataSetBaseInitArgMap.Expand(args)
    d.be = backends.Be().(generators.Generator)
    d.iterMap = make(map[string]*DataIterator)
    d.index = d.be.MakeIndex("dataset")
    d.be.AddGlobalObject(self)
}

func(d *DataSetBase) Be() generators.Generator {
    return d.be
}

func(d *DataSetBase) GetIter(key string) *DataIterator {
    return d.iterMap[key]
}

func(d *DataSetBase) SetIter(key string, iter *DataIterator) {
    d.iterMap[key] = iter
}

func(d *DataSetBase) TrainIter() *DataIterator {
    return d.iterMap["train"]
}

func(d *DataSetBase) ValidIter() *DataIterator {
    return d.iterMap["valid"]
}

func(d *DataSetBase) TestIter() *DataIterator {
    return d.iterMap["test"]
}

func(d *DataSetBase) Index() int {
    return d.index
}

