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

package data

import (
    "fragata/arhat/backends"
    "fragata/arhat/base"
)

//
//    DataIterator
//

type DataIterator struct {
    base.ObjectBase
    symbol string
    shape []int
    dtype base.Dtype
    x backends.Tensor
    y backends.Tensor
}

func NewDataIterator(
        symbol string,
        xdim1 int, 
        ydim1 int, 
        lshape []int,
        dtype base.Dtype,
        name string) *DataIterator {
    d := new(DataIterator)
    d.Init(d, symbol, xdim1, ydim1, lshape, dtype, name)
    return d
}

func(d *DataIterator) Init(
        self base.Object, 
        symbol string,
        xdim1 int, 
        ydim1 int, 
        lshape []int,
        dtype base.Dtype,
        name string) {
    d.ObjectBase.Init(self, base.Args{"name": name})
    be := backends.Be()
    d.symbol = symbol
    if lshape == nil {
        base.Assert(xdim1 != base.IntNone)
        d.shape = []int{xdim1}
    } else {
        base.Assert(xdim1 == base.IntNone)
        d.shape = lshape
        xdim1 = base.IntsProd(lshape)
    }
    bsz := be.Bsz()
    // ACHTUNG: Originally Iobuf (filled with 0)
    x := be.NewTensor([]int{xdim1, bsz}, dtype)
    y := be.NewTensor([]int{ydim1, bsz}, dtype)
    d.x = x
    d.y = y
}

func(d *DataIterator) ClassName() string {
    return "fragata.arhat.generators.data.DataIterator"
}

func(d *DataIterator) Symbol() string {
    return d.symbol
}

func(d *DataIterator) Shape() []int {
    return d.shape
}

func(d *DataIterator) Dtype() base.Dtype {
    return d.dtype
}

func(d *DataIterator) X() backends.Tensor {
    return d.x
}

func(d *DataIterator) Y() backends.Tensor {
    return d.y
}

