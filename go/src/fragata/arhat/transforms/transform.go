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

package transforms

import (
    "fragata/arhat/backends"
    "fragata/arhat/base"
)

//
//    Transform
//

type Transform interface {
    base.Object
    Call(x backends.Tensor) backends.Value
    Bprop(x backends.Tensor) backends.Value
}

//
//    TransformArg
//

type TransformArg struct {
    base.ArgDefBase
}

func NewTransformArg() *TransformArg {
    a := new(TransformArg)
    a.ArgDefBase.Init(true, nil)
    return a
}

func NewTransformArgOpt(defval Transform) *TransformArg {
    a := new(TransformArg)
    a.ArgDefBase.Init(false, defval)
    return a
}

func(a *TransformArg) Expand(v interface{}) (interface{}, bool) {
    if v == nil {
        return v, true
    }
    if _, ok := v.(Transform); !ok {
        return nil, false
    }
    return v, true
}

func ToTransform(v interface{}) Transform {
    if v == nil {
        return nil
    }
    return v.(Transform)
}

//
//    TransformBase
//

type TransformBase struct {
    base.ObjectBase
    isMklop bool
}

var transformBaseInitArgMap = base.ArgMap{
    "name": base.NewAnyArgOpt(""), // passthru
}

func(t *TransformBase) Init(self base.Object, args base.Args) {
    args = transformBaseInitArgMap.Expand(args)
    t.ObjectBase.Init(self, args.Filter([]string{"name"}))
    t.isMklop = false
}

