//
// Copyright 2019-2020 FRAGATA COMPUTER SYSTEMS AG
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

package data

import (
    "fmt"
    "fragata/arhat/backends"
    "fragata/arhat/base"
    "fragata/arhat/generators"
)

//
//    DataWriter
//

type DataWriter interface {
    base.Object
    Symbol() string
    Shape() []int
    Dtype() base.Dtype
    Declare()
    Initialize()
}

//
//    WriterBase
//

type WriterBase struct {
    base.ObjectBase
    be generators.Generator
    index int
    symbol string
    shape []int
    dtype base.Dtype
}

func(w *WriterBase) Init(
        self DataWriter,
        shape []int,
        dtype base.Dtype,
        name string) {
    w.ObjectBase.Init(self, base.Args{"name": name})
    w.be = backends.Be().(generators.Generator)
    w.index = w.be.MakeIndex("datawriter")
    w.symbol = fmt.Sprintf("writer_%d", w.index)
    w.shape = shape
    w.dtype = dtype
    w.be.AddGlobalObject(self)
}

func(w *WriterBase) Symbol() string {
    return w.symbol
}

func(w *WriterBase) Shape() []int {
    return w.shape
}

func(w *WriterBase) Dtype() base.Dtype {
    return w.dtype
}

//
//    ArrayWriter
//

type ArrayWriter struct {
    WriterBase
}

func NewArrayWriter(shape []int, dtype base.Dtype, name string) *ArrayWriter {
    w := new(ArrayWriter)
    w.Init(w, shape, dtype, name)
    return w
}

func(w *ArrayWriter) Init(
        self DataWriter,
        shape []int,
        dtype base.Dtype,
        name string) {
    w.WriterBase.Init(self, shape, dtype, name)
}

func(w *ArrayWriter) ClassName() string {
    return "fragata.arhat.data.ArrayWriter"
}

func(w *ArrayWriter) Declare() {
    w.be.WriteLine("ArrayWriter %s;", w.symbol)
}

func(w *ArrayWriter) Initialize() {
    be := w.be
    bsz := be.Bsz()
    dim1 := base.IntsProd(w.shape)
    be.WriteLine("")
    be.WriteLine("%s.Init(%d, %d, %d);", w.symbol, dim1, bsz, w.dtype.ItemSize())
}

