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
    "math"
)

//
//    Identity
//

type Identity struct {
    TransformBase
}

var identityInitArgMap = base.ArgMap{
    "name": base.NewAnyArgOpt(""), // passthru
}

func NewIdentity(args ...interface{}) *Identity {
    t := new(Identity)
    t.Init(t, base.MakeArgs(args))
    return t
}

func(t *Identity) Init(self base.Object, args base.Args) {
    args = identityInitArgMap.Expand(args)
    t.TransformBase.Init(self, args.Filter([]string{"name"}))
}

func(t *Identity) Call(x backends.Tensor) backends.Value {
    return x
}

func(t *Identity) Bprop(x backends.Tensor) backends.Value {
    return backends.NewInt(1)
}

func(t *Identity) ClassName() string {
    return "arhat.transforms.Identity"
}

//
//    Rectlin
//

type Rectlin struct {
    TransformBase
    slope float64
}

var rectlinInitArgMap = base.ArgMap{
    "name": base.NewAnyArgOpt(""), // passthru
    "slope": base.NewFloatArgOpt(0.0),
}

func NewRectlin(args ...interface{}) *Rectlin {
    t := new(Rectlin)
    t.Init(t, base.MakeArgs(args))
    return t
}

func(t *Rectlin) Init(self base.Object, args base.Args) {
    args = rectlinInitArgMap.Expand(args)
    t.TransformBase.Init(self, args.Filter([]string{"name"}))
    t.slope = args["slope"].(float64)
}

func(t *Rectlin) Call(x backends.Tensor) backends.Value {
    be := backends.Be()
    return be.FpropRelu(nil, x, t.slope)
}

func(t *Rectlin) Bprop(x backends.Tensor) backends.Value {
    be := backends.Be()
    return be.BpropRelu(nil, x, nil, nil, t.slope)
}

func(t *Rectlin) ClassName() string {
    return "arhat.transforms.Rectlin"
}

//
//    Rectlinclip
//

type Rectlinclip struct {
    TransformBase
    slope float64
    xcut float64
}

var rectlinclipInitArgMap = base.ArgMap{
    "name": base.NewAnyArgOpt(""), // passthru
    "slope": base.NewFloatArgOpt(0.0),
    "xcut": base.NewFloatArgOpt(20.0),
}

func NewRectlinclip(args ...interface{}) *Rectlinclip {
    t := new(Rectlinclip)
    t.Init(t, base.MakeArgs(args))
    return t
}

func(t *Rectlinclip) Init(self base.Object, args base.Args) {
    args = rectlinclipInitArgMap.Expand(args)
    t.TransformBase.Init(self, args.Filter([]string{"name"}))
    t.slope = args["slope"].(float64)
    t.xcut = args["xcut"].(float64)
}

func(t *Rectlinclip) Call(x backends.Tensor) backends.Value {
    // minimum(maximum(x, 0) + slope * minimum(x, 0), xcut)
    be := backends.Be()
    zero := be.Float(0.0)
    slope := be.Float(t.slope)
    xcut := be.Float(t.xcut)
    return be.Minimum(be.Maximum(x, zero).Add(slope.Mul(be.Minimum(x, zero))), xcut)
}

func(t *Rectlinclip) Bprop(x backends.Tensor) backends.Value {
    // greater(x, 0) + slope * less(x, 0) * greater(xcut, x)
    be := backends.Be()
    zero := be.Float(0.0)
    slope := be.Float(t.slope)
    xcut := be.Float(t.xcut)
    return be.Greater(x, zero).Add(slope.Mul(be.Less(x, zero).Mul(be.Greater(xcut, x))))
}

func(t *Rectlinclip) ClassName() string {
    return "arhat.transforms.Rectlinclip"
}

//
//    Explin
//

type Explin struct {
    TransformBase
    alpha float64
}

var explinInitArgMap = base.ArgMap{
    "name": base.NewAnyArgOpt(""), // passthru
    "alpha": base.NewFloatArgOpt(1.0),
}

func NewExplin(args ...interface{}) *Explin {
    t := new(Explin)
    t.Init(t, base.MakeArgs(args))
    return t
}

func(t *Explin) Init(self base.Object, args base.Args) {
    args = explinInitArgMap.Expand(args)
    t.TransformBase.Init(self, args.Filter([]string{"name"}))
    t.alpha = args["alpha"].(float64)
}

func(t *Explin) Call(x backends.Tensor) backends.Value {
    // maximum(x, 0) + alpha * (exp(minimum(x, 0)) - 1
    be := backends.Be()
    zero := be.Float(0.0)
    one := be.Float(1.0)
    alpha := be.Float(t.alpha)
    return be.Maximum(x, zero).Add(alpha.Mul(be.Exp(be.Minimum(x, zero)))).Sub(one)
}

func(t *Explin) Bprop(x backends.Tensor) backends.Value {
    // greater(x, 0) + minimum(x, 0) + alpha * less(x, 0)
    be := backends.Be()
    zero := be.Float(0.0)
    alpha := be.Float(t.alpha)
    return be.Greater(x, zero).Add(be.Minimum(x, zero).Add(alpha.Mul(be.Less(x, zero))))
}

func(t *Explin) ClassName() string {
    return "arhat.transforms.Explin"
}

//
//    Normalizer
//

type Normalizer struct {
    TransformBase
    divisor float64
}

var normalizerInitArgMap = base.ArgMap{
    "name": base.NewAnyArgOpt(""), // passthru
    "divisor": base.NewFloatArgOpt(128.0),
}

func NewNormalizer(args ...interface{}) *Normalizer {
    t := new(Normalizer)
    t.Init(t, base.MakeArgs(args))
    return t
}

func(t *Normalizer) Init(self base.Object, args base.Args) {
    args = normalizerInitArgMap.Expand(args)
    t.TransformBase.Init(self, args.Filter([]string{"name"}))
    t.divisor = args["divisor"].(float64)
}

func(t *Normalizer) Call(x backends.Tensor) backends.Value {
    be := backends.Be()
    divisor := be.Float(t.divisor)
    return x.Div(divisor)
}

func(t *Normalizer) Bprop(x backends.Tensor) backends.Value {
    be := backends.Be()
    return be.Float(1.0/t.divisor)
}

func(t *Normalizer) ClassName() string {
    return "arhat.transforms.Normalizer"
}

//
//    Softmax
//

type Softmax struct {
    TransformBase
    axis int
    epsilon float64
}

var softmaxInitArgMap = base.ArgMap{
    "name": base.NewAnyArgOpt(""), // passthru
    "axis": base.NewIntArgOpt(0),
    "epsilon": base.NewFloatArgOpt(math.Ldexp(1.0, -23)), // 2**-23
}

func NewSoftmax(args ...interface{}) *Softmax {
    t := new(Softmax)
    t.Init(t, base.MakeArgs(args))
    return t
}

func(t *Softmax) Init(self base.Object, args base.Args) {
    args = softmaxInitArgMap.Expand(args)
    t.TransformBase.Init(self, args.Filter([]string{"name"}))
    t.axis = args["axis"].(int)
    t.epsilon = args["epsilon"].(float64) // unused
}

func(t *Softmax) Call(x backends.Tensor) backends.Value {
    be := backends.Be()
    return be.FpropSoftmax(x, t.axis)
}

func(t *Softmax) Bprop(x backends.Tensor) backends.Value {
    be := backends.Be()
    return be.Float(1.0)
}

func(t *Softmax) ClassName() string {
    return "arhat.transforms.Softmax"
}

//
//    PixelwiseSoftmax
//

type PixelwiseSoftmax struct {
    TransformBase
    c int
    epsilon float64
}

var pixelwiseSoftmaxInitArgMap = base.ArgMap{
    "name": base.NewAnyArgOpt(""), // passthru
    "c": base.NewIntArg(),
    "epsilon": base.NewFloatArgOpt(math.Ldexp(1.0, -23)), // 2**-23
}

func NewPixelwiseSoftmax(args ...interface{}) *PixelwiseSoftmax {
    t := new(PixelwiseSoftmax)
    t.Init(t, base.MakeArgs(args))
    return t
}

func(t *PixelwiseSoftmax) Init(self base.Object, args base.Args) {
    args = pixelwiseSoftmaxInitArgMap.Expand(args)
    t.TransformBase.Init(self, args.Filter([]string{"name"}))
    t.c = args["c"].(int)
    t.epsilon = args["epsilon"].(float64)
}

func(t *PixelwiseSoftmax) Call(x backends.Tensor) backends.Value {
    // y = x.reshape(c, -1)
    // y[] = (reciprocal(sum(exp(y - max(y, axis=0)), axis=0)) * exp(y - max(y, axis=0)))
    be := backends.Be()
    y := x.Reshape([]int{t.c, -1})
    return y.Assign(
        be.Reciprocal(be.Sum(be.Exp(y.Sub(be.Max(y, 0))), 0)).Mul(y.Sub(be.Max(y, 0))))
}

func(t *PixelwiseSoftmax) Bprop(x backends.Tensor) backends.Value {
    be := backends.Be()
    return be.Float(1.0)
}

func(t *PixelwiseSoftmax) ClassName() string {
    return "arhat.transforms.PixelwiseSoftmax"
}

//
//    Tanh
//

type Tanh struct {
    TransformBase
}

var tanhInitArgMap = base.ArgMap{
    "name": base.NewAnyArgOpt(""), // passthru
}

func NewTanh(args ...interface{}) *Tanh {
    t := new(Tanh)
    t.Init(t, base.MakeArgs(args))
    return t
}

func(t *Tanh) Init(self base.Object, args base.Args) {
    args = tanhInitArgMap.Expand(args)
    t.TransformBase.Init(self, args.Filter([]string{"name"}))
}

func(t *Tanh) Call(x backends.Tensor) backends.Value {
    be := backends.Be()
    return be.Tanh(x)
}

func(t *Tanh) Bprop(x backends.Tensor) backends.Value {
    be := backends.Be()
    one := be.Float(1.0)
    return one.Sub(be.Square(x))
}

func(t *Tanh) ClassName() string {
    return "arhat.transforms.Tanh"
}

//
//    Logistic
//

type Logistic struct {
    TransformBase
    shortcut bool
}

var logisticInitArgMap = base.ArgMap{
    "name": base.NewAnyArgOpt(""), // passthru
    "shortcut": base.NewBoolArgOpt(false),
}

func NewLogistic(args ...interface{}) *Logistic {
    t := new(Logistic)
    t.Init(t, base.MakeArgs(args))
    return t
}

func(t *Logistic) Init(self base.Object, args base.Args) {
    args = logisticInitArgMap.Expand(args)
    t.TransformBase.Init(self, args.Filter([]string{"name"}))
    t.shortcut = args["shortcut"].(bool)
}

func(t *Logistic) Call(x backends.Tensor) backends.Value {
    be := backends.Be()
    return be.Sig(x)
}

func(t *Logistic) Bprop(x backends.Tensor) backends.Value {
    be := backends.Be()
    one := be.Float(1.0)
    if t.shortcut {
        return one
    } else {
        return x.Mul(one.Sub(x))
    }
}

func(t *Logistic) ClassName() string {
    return "arhat.transforms.Logistic"
}

//
//    Sign
//

type Sign struct {
    TransformBase
    inputs backends.Tensor
}

var signInitArgMap = base.ArgMap{
    "name": base.NewAnyArgOpt(""), // passthru
}

func NewSign(args ...interface{}) *Sign {
    t := new(Sign)
    t.Init(t, base.MakeArgs(args))
    return t
}

func(t *Sign) Init(self base.Object, args base.Args) {
    args = signInitArgMap.Expand(args)
    t.TransformBase.Init(self, args.Filter([]string{"name"}))
}

func(t *Sign) Call(x backends.Tensor) backends.Value {
    be := backends.Be()
    // orig: be.array(x.get())
    t.inputs = be.NewTensor(x.Shape(), x.Dtype())
    t.inputs.Copy(x)
    return be.Binarize(x, x, false)
}

func(t *Sign) Bprop(x backends.Tensor) backends.Value {
    // less_equal(absolute(self.inputs), 1)
    be := backends.Be()
    one := be.Float(1.0)
    return be.LessEqual(t.inputs, one)
}

func(t *Sign) ClassName() string {
    return "arhat.transforms.Sign"
}

