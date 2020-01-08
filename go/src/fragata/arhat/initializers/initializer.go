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

package initializers

import (
    "fragata/arhat/backends"
    "fragata/arhat/base"
    "math"
)

//
//    Initializer
//

type Initializer interface {
    base.Object
    Fill(param backends.Tensor)
}

//
//    InitializerArg
//

type InitializerArg struct {
    base.ArgDefBase
}

func NewInitializerArg() *InitializerArg {
    a := new(InitializerArg)
    a.ArgDefBase.Init(true, nil)
    return a
}

func NewInitializerArgOpt(defval Initializer) *InitializerArg {
    a := new(InitializerArg)
    a.ArgDefBase.Init(false, defval)
    return a
}

func(a *InitializerArg) Expand(v interface{}) (interface{}, bool) {
    if v == nil {
        return v, true
    }
    if _, ok := v.(Initializer); !ok {
        return nil, false
    }
    return v, true
}

func ToInitializer(v interface{}) Initializer {
    if v == nil {
        return nil
    }
    return v.(Initializer)
}

//
//    InitializerBase
//

type InitializerBase struct {
    base.ObjectBase
}

//
//    Constant
//

type Constant struct {
    InitializerBase
    val float64
}

var constantInitArgMap = base.ArgMap{
    "val": base.NewFloatArgOpt(0.0),
    "name": base.NewAnyArgOpt(""), // passthru
}

func NewConstant(args ...interface{}) *Constant {
    n := new(Constant)
    n.Init(n, base.MakeArgs(args))
    return n
}

func(n *Constant) Init(self base.Object, args base.Args) {
    args = constantInitArgMap.Expand(args)
    n.InitializerBase.Init(self, args.Filter([]string{"name"}))
    n.val = args["val"].(float64)
}

func(n *Constant) ClassName() string {
    return "arhat.initializers.Constant"
}

func(n *Constant) Fill(param backends.Tensor) {
    param.Assign(backends.Be().Float(n.val))
}

//
//    Uniform
//

type Uniform struct {
    InitializerBase
    low float64
    high float64
}

var uniformInitArgMap = base.ArgMap{
    "low": base.NewFloatArgOpt(0.0),
    "high": base.NewFloatArgOpt(1.0),
    "name": base.NewAnyArgOpt(""), // passthru
}

func NewUniform(args ...interface{}) *Uniform {
    n := new(Uniform)
    n.Init(n, base.MakeArgs(args))
    return n
}

func(n *Uniform) Init(self base.Object, args base.Args) {
    args = uniformInitArgMap.Expand(args)
    n.InitializerBase.Init(self, args.Filter([]string{"name"}))
    n.low = args["low"].(float64)
    n.high = args["high"].(float64)
}

func(n *Uniform) ClassName() string {
    return "arhat.initializers.Uniform"
}

func(n *Uniform) Fill(param backends.Tensor) {
    backends.Be().RngUniform(param, n.low, n.high, param.Shape())
}

//
//    Gaussian
//

type Gaussian struct {
    InitializerBase
    loc float64
    scale float64
}

var gaussianInitArgMap = base.ArgMap{
    "loc": base.NewFloatArgOpt(0.0),
    "scale": base.NewFloatArgOpt(1.0),
    "name": base.NewAnyArgOpt(""), // passthru
}

func NewGaussian(args ...interface{}) *Gaussian {
    n := new(Gaussian)
    n.Init(n, base.MakeArgs(args))
    return n
}

func(n *Gaussian) Init(self base.Object, args base.Args) {
    args = gaussianInitArgMap.Expand(args)
    n.InitializerBase.Init(self, args.Filter([]string{"name"}))
    n.loc = args["loc"].(float64)
    n.scale = args["scale"].(float64)
}

func(n *Gaussian) ClassName() string {
    return "arhat.initializers.Gaussian"
}

func(n *Gaussian) Fill(param backends.Tensor) {
    backends.Be().RngNormal(param, n.loc, n.scale, param.Shape())
}


//
//    GlorotUniform
//

type GlorotUniform struct {
    InitializerBase
}

var glorotUniformInitArgMap = base.ArgMap{
    "name": base.NewAnyArgOpt(""), // passthru
}

func NewGlorotUniform(args ...interface{}) *GlorotUniform {
    n := new(GlorotUniform)
    n.Init(n, base.MakeArgs(args))
    return n
}

func(n *GlorotUniform) Init(self base.Object, args base.Args) {
    args = glorotUniformInitArgMap.Expand(args)
    n.InitializerBase.Init(self, args.Filter([]string{"name"}))
}

func(n *GlorotUniform) ClassName() string {
    return "arhat.initializers.GlorotUniform"
}

func(n *GlorotUniform) Fill(param backends.Tensor) {
    shape := param.Shape()
    k := math.Sqrt(6.0/float64(shape[0]+shape[1]))
    backends.Be().RngUniform(param, -k, k, shape)
}

//
//    Xavier
//

type Xavier struct {
    InitializerBase
    local bool
}

var xavierInitArgMap = base.ArgMap{
    "local": base.NewBoolArgOpt(true),
    "name": base.NewAnyArgOpt(""), // passthru
}

func NewXavier(args ...interface{}) *Xavier {
    n := new(Xavier)
    n.Init(n, base.MakeArgs(args))
    return n
}

func(n *Xavier) Init(self base.Object, args base.Args) {
    args = xavierInitArgMap.Expand(args)
    n.InitializerBase.Init(self, args.Filter([]string{"name"}))
    n.local = args["local"].(bool)
}

func(n *Xavier) ClassName() string {
    return "arhat.initializers.Xavier"
}

func(n *Xavier) Fill(param backends.Tensor) {
    shape := param.Shape()
    var fanIn int
    if n.local {
        fanIn = shape[0]
    } else {
        fanIn = shape[1]
    }
    scale := math.Sqrt(3.0/float64(fanIn))
    backends.Be().RngUniform(param, -scale, scale, shape)
}

//
//    Kaiming
//

type Kaiming struct {
    InitializerBase
    local bool
}

var kaimingInitArgMap = base.ArgMap{
    "local": base.NewBoolArgOpt(true),
    "name": base.NewAnyArgOpt(""), // passthru
}

func NewKaiming(args ...interface{}) *Kaiming {
    n := new(Kaiming)
    n.Init(n, base.MakeArgs(args))
    return n
}

func(n *Kaiming) Init(self base.Object, args base.Args) {
    args = kaimingInitArgMap.Expand(args)
    n.InitializerBase.Init(self, args.Filter([]string{"name"}))
    n.local = args["local"].(bool)
}

func(n *Kaiming) ClassName() string {
    return "arhat.initializers.Kaiming"
}

func(n *Kaiming) Fill(param backends.Tensor) {
    shape := param.Shape()
    var fanIn int
    if n.local {
        fanIn = shape[0]
    } else {
        fanIn = shape[1]
    }
    scale := math.Sqrt(2.0/float64(fanIn))
    backends.Be().RngNormal(param, 0.0, scale, shape)
}

//
//    Identity
//

type Identity struct {
    InitializerBase
    local bool
}

var identityInitArgMap = base.ArgMap{
    "local": base.NewBoolArgOpt(true),
    "name": base.NewAnyArgOpt(""), // passthru
}

func NewIdentity(args ...interface{}) *Identity {
    n := new(Identity)
    n.Init(n, base.MakeArgs(args))
    return n
}

func(n *Identity) Init(self base.Object, args base.Args) {
    args = identityInitArgMap.Expand(args)
    n.InitializerBase.Init(self, args.Filter([]string{"name"}))
    n.local = args["local"].(bool)
}

func(n *Identity) ClassName() string {
    return "arhat.initializers.Identity"
}

func(n *Identity) Fill(param backends.Tensor) {
    // TODO
    base.NotImplementedError()
}

//
//    Orthonormal
//

type Orthonormal struct {
    InitializerBase
    loc float64
    scale float64
}

var orthonormalInitArgMap = base.ArgMap{
    "scale": base.NewFloatArgOpt(1.1),
    "name": base.NewAnyArgOpt(""), // passthru
}

func NewOrthonormal(args ...interface{}) *Orthonormal {
    n := new(Orthonormal)
    n.Init(n, base.MakeArgs(args))
    return n
}

func(n *Orthonormal) Init(self base.Object, args base.Args) {
    args = orthonormalInitArgMap.Expand(args)
    n.InitializerBase.Init(self, args.Filter([]string{"name"}))
    n.scale = args["scale"].(float64)
}

func(n *Orthonormal) ClassName() string {
    return "arhat.initializers.Orthonormal"
}

func(n *Orthonormal) Fill(param backends.Tensor) {
    // TODO
    base.NotImplementedError()
}

