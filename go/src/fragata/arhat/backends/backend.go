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

package backends

import (
    "fmt"
    "fragata/arhat/base"
)

//
//    Transform
//
//    Replicated transforms.Transform to avoid circular imports
//

type Transform interface {
    base.Object
    Call(x Tensor) Value
    Bprop(x Tensor) Value
}

//
//    Op
//

type Op int

const (
    Abs Op = iota
    Add
    Argmax
    Argmin
    Assign // used only in stacks
    Binarize
    Div
    Dot
    Eq
    Exp
    Exp2
    Finite
    Ge
    Gt
    Le
    Log
    Log2
    Lt
    Min
    Minimum
    Max
    Maximum
    Mul
    Ne
    Neg
    Onehot
    Pow
    Rand
    Rint
    Safelog
    Shift
    Sig
    Sig2
    Sgn
    Sqr
    Sqrt
    Sub
    Sum
    Tanh
    Tanh2
    Transpose
    numOps
)

var opString = [numOps]string{
    Abs: "abs",
    Add: "add",
    Argmax: "argmax",
    Argmin: "argmin",
    Assign: "assign",
    Binarize: "binarize",
    Div: "div",
    Dot: "dot",
    Eq: "eq",
    Exp: "exp",
    Exp2: "exp2",
    Finite: "finite",
    Ge: "ge",
    Gt: "gt",
    Le: "le",
    Log: "log",
    Log2: "log2",
    Lt: "lt",
    Min: "min",
    Minimum: "minimum",
    Max: "max",
    Maximum: "maximum",
    Mul: "mul",
    Ne: "ne",
    Neg: "neg",
    Onehot: "onehot",
    Pow: "pow",
    Rand: "rand",
    Rint: "rint",
    Safelog: "safelog",
    Shift: "shift",
    Sig: "sig",
    Sig2: "sig2",
    Sgn: "sgn",
    Sqr: "sqr",
    Sqrt: "sqrt",
    Sub: "sub",
    Sum: "sum",
    Tanh: "tahn",
    Tanh2: "tahn2",
    Transpose: "transpose",
}

func(op Op) String() string {
    return opString[op]
}

const (
    zeroOperandOps uint8 = 1 << iota
    unaryOps
    binaryOps
    reductionOps
    floatOps
    ewOps
)

var opCollection = [numOps]uint8 {
    Abs: unaryOps | floatOps | ewOps,
    Add: binaryOps | floatOps | ewOps,
    Argmax: reductionOps,
    Argmin: reductionOps,
    Assign: binaryOps | floatOps | ewOps,
    Binarize: unaryOps | floatOps | ewOps,
    Div: binaryOps | floatOps | ewOps,
    Dot: binaryOps | floatOps,
    Eq: binaryOps | floatOps | ewOps,
    Exp: unaryOps | floatOps | ewOps,
    Exp2: unaryOps | floatOps | ewOps,
    Finite: unaryOps | floatOps | ewOps,
    Ge: binaryOps | floatOps | ewOps,
    Gt: binaryOps | floatOps | ewOps,
    Le: binaryOps | floatOps | ewOps,
    Log: unaryOps | floatOps | ewOps,
    Log2: unaryOps | floatOps | ewOps,
    Lt: binaryOps | floatOps | ewOps,
    Min: reductionOps,
    Minimum: binaryOps | floatOps | ewOps,
    Max: reductionOps,
    Maximum: binaryOps | floatOps | ewOps,
    Mul: binaryOps | floatOps | ewOps,
    Ne: binaryOps | floatOps | ewOps,
    Neg: unaryOps | floatOps | ewOps,
    Onehot: zeroOperandOps | floatOps | ewOps,
    Pow: binaryOps | floatOps | ewOps,
    Rand: zeroOperandOps | floatOps | ewOps,
    Rint: unaryOps | floatOps | ewOps,
    Safelog: unaryOps | floatOps | ewOps,
    Shift: binaryOps | floatOps | ewOps,
    Sig: unaryOps | floatOps | ewOps,
    Sig2: unaryOps | floatOps | ewOps,
    Sgn: unaryOps | floatOps | ewOps,
    Sqr: unaryOps | floatOps | ewOps,
    Sqrt: unaryOps | floatOps | ewOps,
    Sub: binaryOps | floatOps | ewOps,
    Sum: reductionOps,
    Tanh: unaryOps | floatOps | ewOps,
    Tanh2: unaryOps | floatOps | ewOps,
    Transpose: unaryOps | floatOps,
}

func IsZeroOperandOp(op Op) bool {
    return (opCollection[op] & zeroOperandOps) != 0
}

func IsUnaryOp(op Op) bool {
    return (opCollection[op] & unaryOps) != 0
}

func IsBinaryOp(op Op) bool {
    return (opCollection[op] & binaryOps) != 0
}

func IsReductionOp(op Op) bool {
    return (opCollection[op] & reductionOps) != 0
}

func IsFloatOp(op Op) bool {
    return (opCollection[op] & floatOps) != 0
}

func IsEwOp(op Op) bool {
    return (opCollection[op] & ewOps) != 0
}

//
//    Value
//

type Value interface {
    String() string
    Shape() []int
    Add(other Value) *OpTreeNode
    Sub(other Value) *OpTreeNode
    Mul(other Value) *OpTreeNode
    Div(other Value) *OpTreeNode
    Pow(other Value) *OpTreeNode
    Eq(other Value) *OpTreeNode
    Ne(other Value) *OpTreeNode
    Lt(other Value) *OpTreeNode
    Le(other Value) *OpTreeNode
    Gt(other Value) *OpTreeNode
    Ge(other Value) *OpTreeNode
    Abs() *OpTreeNode
    Neg() *OpTreeNode
}

//
//    ValueBase
//

type ValueBase struct { 
    self Value
}

func(x *ValueBase) Init(self Value) {
    x.self = self
}

func(x *ValueBase) Self() Value {
    return x.self
}

func(x *ValueBase) Shape() []int {
    // scalar
    return []int{1, 1}
}

func(x *ValueBase) Add(other Value) *OpTreeNode {
    return BuildOpTreeNode(Add, x.self, other)
}

func(x *ValueBase) Sub(other Value) *OpTreeNode {
    return BuildOpTreeNode(Sub, x.self, other)
}

func(x *ValueBase) Mul(other Value) *OpTreeNode {
    return BuildOpTreeNode(Mul, x.self, other)
}

func(x *ValueBase) Div(other Value) *OpTreeNode {
    return BuildOpTreeNode(Div, x.self, other)
}

func(x *ValueBase) Pow(other Value) *OpTreeNode {
    return BuildOpTreeNode(Pow, x.self, other)
}

func(x *ValueBase) Eq(other Value) *OpTreeNode {
    return BuildOpTreeNode(Eq, x.self, other)
}

func(x *ValueBase) Ne(other Value) *OpTreeNode {
    return BuildOpTreeNode(Ne, x.self, other)
}

func(x *ValueBase) Lt(other Value) *OpTreeNode {
    return BuildOpTreeNode(Lt, x.self, other)
}

func(x *ValueBase) Le(other Value) *OpTreeNode {
    return BuildOpTreeNode(Le, x.self, other)
}

func(x *ValueBase) Gt(other Value) *OpTreeNode {
    return BuildOpTreeNode(Gt, x.self, other)
}

func(x *ValueBase) Ge(other Value) *OpTreeNode {
    return BuildOpTreeNode(Ge, x.self, other)
}

func(x *ValueBase) Abs() *OpTreeNode {
    return BuildOpTreeNode(Abs, x.self, nil)
}

func(x *ValueBase) Neg() *OpTreeNode {
    return BuildOpTreeNode(Neg, x.self, nil)
}

//
//    Int
//

type Int struct {
    ValueBase
    v int
}

func NewInt(v int) *Int {
    x := new(Int)
    x.Init(x, v)
    return x
}

func(x *Int) Init(self Value, v int) {
    x.ValueBase.Init(self)
    x.v = v
}

func(x *Int) Value() int {
    return x.v
}

func(x *Int) String() string {
    return fmt.Sprintf("int(%d)", x.v)
}

//
//    Float
//

type Float struct {
    ValueBase
    v float64
}

func NewFloat(v float64) *Float {
    x := new(Float)
    x.Init(x, v)
    return x
}

func(x *Float) Init(self Value, v float64) {
    x.ValueBase.Init(self)
    x.v = v
}

func(x *Float) Value() float64 {
    return x.v
}

func(x *Float) String() string {
    return fmt.Sprintf("float(%g)", x.v)
}

//
//    Slice
//

type Ellipsis struct { }

type SliceItem struct {
    start int
    stop int
    step int
}

func NewSliceItem(start int, stop int, step int) *SliceItem {
    return &SliceItem{start, stop, step}
}

func(s *SliceItem) Indices(length int) (int, int, int) {
    step := s.step
    if step == base.IntNone {
        step = 1
    }
    if step == 0 {
        base.ValueError("slice step cannot be zero")
    }

    defStart := 0
    defStop := length
    if step < 0 {
        defStart = length - 1
        defStop = -1
    }

    start := adjustBound(s.start, defStart, step, length)
    stop := adjustBound(s.stop, defStop, step, length)

    return start, stop, step
}

func adjustBound(bound int, def int, step int, length int) int {
    if bound == base.IntNone {
        bound = def
    } else {
        if bound < 0 {
            bound += length
        }
        if bound < 0 {
            bound = 0
            if step < 0 {
                bound = -1
            }
        }
        if bound > length {
            bound = length
            if step < 0 {
                bound = length - 1
            }
        }
    }
    return bound
}

type Slice struct {
    items []interface{}
}

func(s *Slice) Len() int {
    return len(s.items)
}

func(s *Slice) Item(index int) interface{} {
    return s.items[index]
}

func(s *Slice) AppendItem(item interface{}) {
    s.items = append(s.items, item)
}

func(s *Slice) IsNone() bool {
    if len(s.items) == 1 {
        if x, ok := s.items[0].(*SliceItem); ok {
            return (x.start == base.IntNone && x.stop == base.IntNone && x.step == base.IntNone)
        }
    }
    return false
}

func MakeSlice(items ...interface{}) Slice {
    var s Slice
    for _, x := range items {
        var y interface{}
        switch v := x.(type) {
        case nil:
            y = NewSliceItem(base.IntNone, base.IntNone, base.IntNone)
        case []int:
            start := base.IntNone
            stop := base.IntNone
            step := base.IntNone
            switch len(v) {
            case 0:
                // ok
            case 1:
                start = v[0]
            case 2:
                start = v[0]
                stop = v[1]
            case 3:
                start = v[0]
                stop = v[1]
                step = v[2]
            default:
                base.ValueError("Invalid slice specification")
            }
            y = NewSliceItem(start, stop, step)
        case int, Ellipsis, Tensor:
            y = x
        default:
            base.TypeError("invalid slice item type")
        }
        s.AppendItem(y)
    }
    return s
}

//
//    Tensor
//

type Tensor interface {
    Value
    Backend() Backend
    Size() int
    Dtype() base.Dtype
    Name() string
    SetName(name string)
    SetPersistValues(persistValues bool)
    Base() Tensor
    Len() int
    SetItem(index Slice, value Tensor)
    GetItem(index Slice) Tensor
    Assign(value Value) Tensor
    GetScalar() Value
    Take(indices Tensor, axis int) Tensor
    Fill(value interface{}) Tensor
    Copy(x Tensor) Tensor
    Reshape(shape []int) Tensor
    T() Tensor
    Share(shape []int, dtype base.Dtype, name string) Tensor
    IsContiguous() bool
}

//
//    TensorBase
//

type TensorBase struct {
    ValueBase
    backend Backend
    shape []int
    dtype base.Dtype
    name string
    persistValues bool
    minDims int
}

func(x *TensorBase) Init(
        self Value,
        backend Backend, 
        shape []int,
        dtype base.Dtype,
        name string,
        persistValues bool) {
    x.ValueBase.Init(self)
    x.backend = backend
    x.shape = shape
    x.dtype = dtype
    x.name = name
    x.persistValues = persistValues
    x.minDims = 2
//    x.base = nil
}

func(x *TensorBase) Backend() Backend {
    return x.backend
}

func(x *TensorBase) Shape() []int {
    return x.shape
}

func(x *TensorBase) Dtype() base.Dtype {
    return x.dtype
}

func(x *TensorBase) Name() string {
    return x.name
}

func(x *TensorBase) SetName(name string) {
    x.name = name
}

func(x *TensorBase) SetPersistValues(persistValues bool) {
    x.persistValues = persistValues
}

func(x *TensorBase) MinDims() int {
    return x.minDims
}

func(x *TensorBase) String() string {
    return fmt.Sprintf("tensor(%s)", x.name)
}

//
//    Layer
//

type Layer interface {
    N() int
    SizeI() int
    SizeF() int
    SizeO() int
}

//
//    ConvLayerBase
//

type ConvLayerBase interface {
    Layer
    C() int
    K() int
    M() int
    P() int
    Q() int
    NCK() []int
    TRS() []int
    DHW() []int
    MPQ() []int
    Padding() []int
    Strides() []int
    DimI() []int
    DimF() []int
    DimO() []int
    DimI2() []int
    DimF2() []int
    DimO2() []int
    DimS() []int
    NOut() int
}

//
//    ConvLayer
//

type ConvParams struct {
    N int
    C int 
    K int
    D int // 3D parameter
    H int 
    W int
    T int // 3D parameter
    R int 
    S int
    PadD int
    PadH int 
    PadW int
    StrD int 
    StrH int 
    StrW int
    DilD int 
    DilH int 
    DilW int
}

func (a *ConvParams) Init() {
    a.N = base.IntNone
    a.C = base.IntNone
    a.K = base.IntNone
    a.D = base.IntNone
    a.H = base.IntNone
    a.W = base.IntNone
    a.T = base.IntNone
    a.R = base.IntNone
    a.S = base.IntNone
    a.PadD = base.IntNone
    a.PadH = base.IntNone
    a.PadW = base.IntNone
    a.StrD = base.IntNone
    a.StrH = base.IntNone
    a.StrW = base.IntNone
    a.DilD = base.IntNone
    a.DilH = base.IntNone
    a.DilW = base.IntNone
}

func (a *ConvParams) Resolve() {
    base.Assert(
        a.N != base.IntNone &&
        a.C != base.IntNone &&
        a.K != base.IntNone)
    a.D = base.ResolveInt(a.D, 1)
    a.H = base.ResolveInt(a.H, 1)
    a.W = base.ResolveInt(a.W, 1)
    a.T = base.ResolveInt(a.T, 1)
    a.R = base.ResolveInt(a.R, 1)
    a.S = base.ResolveInt(a.S, 1)
    a.PadD = base.ResolveInt(a.PadD, 0)
    a.PadH = base.ResolveInt(a.PadH, 0)
    a.PadW = base.ResolveInt(a.PadW, 0)
    a.StrD = base.ResolveInt(a.StrD, 1)
    a.StrH = base.ResolveInt(a.StrH, 1)
    a.StrW = base.ResolveInt(a.StrW, 1)
    a.DilD = base.ResolveInt(a.DilD, 1)
    a.DilH = base.ResolveInt(a.DilH, 1)
    a.DilW = base.ResolveInt(a.DilW, 1)
}

type ConvLayer interface {
    ConvLayerBase
}

//
//    DeconvLayer
//

type DeconvParams struct {
    N int
    C int 
    K int
    M int
    P int
    Q int
    T int
    R int 
    S int
    PadD int
    PadH int 
    PadW int
    StrD int 
    StrH int 
    StrW int
    DilD int 
    DilH int 
    DilW int
}

func (a *DeconvParams) Init() {
    a.N = base.IntNone
    a.C = base.IntNone
    a.K = base.IntNone
    a.M = base.IntNone
    a.P = base.IntNone
    a.Q = base.IntNone
    a.T = base.IntNone
    a.R = base.IntNone
    a.S = base.IntNone
    a.PadD = base.IntNone
    a.PadH = base.IntNone
    a.PadW = base.IntNone
    a.StrD = base.IntNone
    a.StrH = base.IntNone
    a.StrW = base.IntNone
    a.DilD = base.IntNone
    a.DilH = base.IntNone
    a.DilW = base.IntNone
}

func (a *DeconvParams) Resolve() {
    base.Assert(
        a.N != base.IntNone &&
        a.C != base.IntNone &&
        a.K != base.IntNone &&
        a.M != base.IntNone &&
        a.P != base.IntNone &&
        a.Q != base.IntNone)
    a.T = base.ResolveInt(a.T, 1)
    a.R = base.ResolveInt(a.R, 1)
    a.S = base.ResolveInt(a.S, 1)
    a.PadD = base.ResolveInt(a.PadD, 0)
    a.PadH = base.ResolveInt(a.PadH, 0)
    a.PadW = base.ResolveInt(a.PadW, 0)
    a.StrD = base.ResolveInt(a.StrD, 1)
    a.StrH = base.ResolveInt(a.StrH, 1)
    a.StrW = base.ResolveInt(a.StrW, 1)
    a.DilD = base.ResolveInt(a.DilD, 1)
    a.DilH = base.ResolveInt(a.DilH, 1)
    a.DilW = base.ResolveInt(a.DilW, 1)
}

type DeconvLayer interface {
    ConvLayerBase
}

//
//    LrnLayer
//

type LrnParams struct {
    N int
    C int
    D int
    H int
    W int
    J int
}

func(a *LrnParams) Init() {
    a.N = base.IntNone
    a.C = base.IntNone
    a.D = base.IntNone
    a.H = base.IntNone
    a.W = base.IntNone
    a.J = base.IntNone
}

func(a *LrnParams) Resolve() {
    base.Assert(
        a.N != base.IntNone &&
        a.C != base.IntNone)
    a.D = base.ResolveInt(a.D, 1)
    a.H = base.ResolveInt(a.H, 1)
    a.W = base.ResolveInt(a.W, 1)
    a.J = base.ResolveInt(a.J, 1)
}

type LrnLayer interface {
    Layer
}

//
//    PoolLayer
//

type PoolOp int

const (
    PoolOpMax PoolOp = iota
    PoolOpAvg
    PoolOpL2
    PoolOpLrn
)

var poolOpString = map[PoolOp]string{
    PoolOpMax: "max",
    PoolOpAvg: "avg",
    PoolOpL2: "l2",
    PoolOpLrn: "lrn",
}

func(op PoolOp) String() string {
    return poolOpString[op]
}

type PoolParams struct {
    Op PoolOp
    N int
    C int
    D int
    H int
    W int
    J int
    T int
    R int
    S int
    PadC int
    PadD int
    PadH int
    PadW int
    StrC int
    StrD int
    StrH int
    StrW int
}

func(a *PoolParams) Init(op PoolOp) {
    a.Op = op
    a.N = base.IntNone
    a.C = base.IntNone
    a.D = base.IntNone
    a.H = base.IntNone
    a.W = base.IntNone
    a.J = base.IntNone
    a.T = base.IntNone
    a.R = base.IntNone
    a.S = base.IntNone
    a.PadH = base.IntNone
    a.PadW = base.IntNone
    a.PadD = base.IntNone
    a.PadC = base.IntNone
    a.StrH = base.IntNone
    a.StrW = base.IntNone
    a.StrD = base.IntNone
    a.StrC = base.IntNone
}

func(a *PoolParams) Resolve() {
    // strides may be None
    base.Assert(
        a.N != base.IntNone &&
        a.C != base.IntNone)
    a.D = base.ResolveInt(a.D, 1)
    a.H = base.ResolveInt(a.H, 1)
    a.W = base.ResolveInt(a.W, 1)
    a.J = base.ResolveInt(a.J, 1)
    a.T = base.ResolveInt(a.T, 1)
    a.R = base.ResolveInt(a.R, 1)
    a.S = base.ResolveInt(a.S, 1)
    a.PadH = base.ResolveInt(a.PadH, 0)
    a.PadW = base.ResolveInt(a.PadW, 0)
    a.PadD = base.ResolveInt(a.PadD, 0)
    a.PadC = base.ResolveInt(a.PadC, 0)
}

type PoolLayer interface {
    Layer
    Op() PoolOp
    C() int
    K() int
    M() int
    P() int
    Q() int
    JTRS() []int
    DHW() []int
    MPQ() []int
    Padding() []int
    Strides() []int
    DimI() []int
    DimO() []int
    DimF2() []int
    DimI2() []int
    DimO2() []int
    NOut() int
}

//
//    BatcNormLayer
//

type BatchNormLayer interface {
    Layer
}

//
//    Backend
//

type Backend interface {
    Id() int
    Bsz() int
    SetBsz(bsz int)
    DefaultDtype() base.Dtype
    OutputDim(x int, s int, padding int, strides int, pooling bool, dilation int) int
    CheckCaffeCompat() bool
    Iobuf(
        dim0 []int, 
        x Tensor, 
        dtype base.Dtype, 
        name string,
        persistValues bool, 
        shared Tensor) Tensor
    SharedIobufSize(shape []int) int
    NewTensor(shape []int, dtype base.Dtype) Tensor
    Int(v int) *Int
    Float(v float64) *Float
    Dot(x Value, y Value) Value
    Add(x Value, y Value) Value
    Subtract(x Value, y Value) Value
    Multiply(x Value, y Value) Value
    Divide(x Value, y Value) Value
    TrueDivide(x Value, y Value) Value
    Power(x Value, y Value) Value
    Reciprocal(x Value) Value
    Negative(x Value) Value
    Sgn(x Value) Value
    Absolute(x Value) Value
    Fabs(x Value) Value
    Sqrt(x Value) Value
    Square(x Value) Value
    Exp(x Value) Value
    Exp2(x Value) Value
    Safelog(x Value) Value
    Log(x Value) Value
    Log2(x Value) Value
    Sig(x Value) Value
    Sig2(x Value) Value
    Tanh(x Value) Value
    Tanh2(x Value) Value
    Finite(x Value) Value
    Rint(x Value) Value
    Equal(x Value, y Value) Value
    NotEqual(x Value, y Value) Value
    Less(x Value, y Value) Value
    LessEqual(x Value, y Value) Value
    Greater(x Value, y Value) Value
    GreaterEqual(x Value, y Value) Value
    Maximum(x Value, y Value) Value
    Minimum(x Value, y Value) Value
    Shift(x Value, y Value, value bool) Value
    Clip(x Value, xMin Value, xMax Value) Value
    Sum(x Value, axis int) Value
    Max(x Value, axis int) Value
    Min(x Value, axis int) Value
    Argmax(x Value, axis int) Value
    Argmin(x Value, axis int) Value
    Mean(x Value, axis int) Value
    Var(x Value, axis int, binary bool) Value
    Std(x Value, axis int) Value
    Take(x Tensor, indices Tensor, axis int) Tensor
    Onehot(indices Tensor, axis int) Value
    // rng methods
    RngNormal(out Tensor, loc float64, scale float64, size []int)
    RngUniform(out Tensor, low float64, high float64, size []int)
    // Not in original backends.Backend
    CompoundDot(
        x Tensor,
        y Tensor,
        z Tensor,
        alpha float64,
        beta float64,
        relu bool,
        bsum Tensor) Tensor
    Binarize(ary Tensor, out Tensor, stochastic bool) Tensor
    MakeBinaryMask(out Tensor, keepThresh float64)
    NewConvLayer(dtype base.Dtype, params *ConvParams) ConvLayer
    NewDeconvLayer(dtype base.Dtype, params *DeconvParams) DeconvLayer
    // SKIPPED: repeat (gpu only), layerOp
    FpropConv(
        layer ConvLayerBase,
        i Tensor,
        f Tensor,
        o Tensor,
        x Tensor,
        bias Tensor,
        bsum Tensor,
        alpha float64,
        beta float64,
        relu bool,
        brelu bool,
        slope float64)
    // SKIPPED: repeat (gpu only), layerOp
    BpropConv(
        layer ConvLayerBase,
        f Tensor,
        e Tensor,
        gradI Tensor,
        x Tensor,
        bias Tensor,
        bsum Tensor,
        alpha float64,
        beta float64,
        relu bool,
        brelu bool,
        slope float64)
    // SKIPPED: repeat (gpu only), layerOp
    UpdateConv(
        layer ConvLayerBase,
        i Tensor,
        e Tensor,
        gradF Tensor,
        alpha float64,
        beta float64,
        gradBias Tensor)
    NewLrnLayer(dtype base.Dtype, params *LrnParams) LrnLayer
    // SKIPPED: repeat (gpu only)
    FpropLrn(
        layer LrnLayer, 
        i Tensor,
        o Tensor,
        denom Tensor,
        alpha float64,
        beta float64,
        ascale float64,
        bpower float64)
    // SKIPPED: repeat (gpu only)
    BpropLrn(
        layer LrnLayer, 
        i Tensor,
        o Tensor,
        e Tensor,
        delta Tensor,
        denom Tensor,
        alpha float64,
        beta float64,
        ascale float64,
        bpower float64)
    NewPoolLayer(dtype base.Dtype, params *PoolParams) PoolLayer
    // SKIPPED: repeat (gpu only)
    FpropPool(
        layer PoolLayer, 
        i Tensor, 
        o Tensor, 
        argmax Tensor, 
        alpha float64, 
        beta float64)
    // SKIPPED: repeat (gpu only)
    BpropPool(
        layer PoolLayer, 
        i Tensor, 
        o Tensor, 
        argmax Tensor, 
        alpha float64, 
        beta float64)
    NewReluLayer() Layer
    FpropRelu(
        layer Layer,
        x Tensor, 
        slope float64) Value
    BpropRelu(
        layer Layer,
        x Tensor,
        errors Tensor,
        deltas Tensor,
        slope float64) Value
    NewBatchNormLayer(inShape []int) BatchNormLayer
    // SKIPPED: threads, repeat (both gpu only)
    CompoundFpropBn(
        x Tensor,
        xsum Tensor,
        xvar Tensor,
        gmean Tensor,
        gvar Tensor,
        gamma Tensor,
        beta Tensor,
        y Tensor,
        eps float64,
        rho float64,
        computeBatchSum bool,
        accumbeta float64,
        relu bool,
        binary bool,
        inference bool,
        outputs Tensor,
        layer BatchNormLayer)
    // SKIPPED: threads, repeat (both gpu only)
    CompoundBpropBn(
        deltaOut Tensor,
        gradGamma Tensor,
        gradBeta Tensor,
        deltaIn Tensor,
        x Tensor,
        xsum Tensor,
        xvar Tensor,
        gamma Tensor,
        eps float64,
        binary bool,
        layer BatchNormLayer)
    FpropSoftmax(x Value, axis int) Value
    FpropTransform(
        nglayer Layer, 
        transform Transform, 
        inputs Tensor, 
        outputs Tensor, 
        relu bool)
    BpropTransform(
        nglayer Layer,
        transform Transform,
        outputs Tensor,
        errors Tensor,
        deltas Tensor,
        relu bool)
    FpropSkipNode(x Tensor, y Tensor, beta float64)
    BpropSkipNode(errors Tensor, deltas Tensor, alpha float64, beta float64)
}

var be Backend

func Be() Backend {
    return be
}

func SetBe(backend Backend) {
    be = backend
}

var nextBackendId int = 0

//
//    CompatMode
//

type CompatMode int

const (
    CompatModeNone = iota
    CompatModeCaffe
)

//
//    BackendBase
//

type BackendBase struct {
    self Backend
    id int
    defaultDtype base.Dtype
    rngSeed int
    // TODO: rng
    bsz int
    compatMode CompatMode
    minDims int
    deterministic bool
}

func(b *BackendBase) Init(
        self Backend, rngSeed int, defaultDtype base.Dtype, compatMode CompatMode) {
    b.self = self
    b.id = nextBackendId
    nextBackendId++

    // dtype
    b.defaultDtype = base.ResolveDtype(defaultDtype, base.Float32)

    // use RandomState instead of seed
    b.rngSeed = rngSeed
/* TODO
    b.rng = self.GenRng(rngSeed)
*/

    // batch size
    b.bsz = base.IntNone
    b.minDims = 2

    b.compatMode = compatMode

    b.deterministic = (b.rngSeed != base.IntNone)
}

func(b *BackendBase) Self() Backend {
    return b.self
}

func(b *BackendBase) Id() int {
    return b.id
}

func(b *BackendBase) DefaultDtype() base.Dtype {
    return b.defaultDtype
}

func(b *BackendBase) RngSeed() int {
    return b.rngSeed
}

func(b *BackendBase) Bsz() int {
    return b.bsz
}

func(b *BackendBase) SetBsz(bsz int) {
    b.bsz = bsz
}

func(b *BackendBase) Deterministic() bool {
    return b.deterministic
}

func(b *BackendBase) OutputDim(
        x int, s int, padding int, strides int, pooling bool, dilation int) int {
    s = dilation * (s - 1) + 1

    var size int
    if b.CheckCaffeCompat() && pooling {
        // TODO: Verify this
        size = (x - s + 2 * padding + strides - 1) / strides + 1
        if padding > 0 && (size - 1) * strides >= x + padding {
            // decrement size if last pooling op is completely in padding
            size--
        }
    } else {
        // normal arhat output size determination
        size = (x - s + 2 * padding) / strides + 1
    }

    if pooling && padding >= 5 {
        base.ValueError("Padding dim %d incompatible with filter size %d", padding, 5)
    }

    return size
}

func(b *BackendBase) CheckCaffeCompat() bool {
    return (b.compatMode == CompatModeCaffe)
}

func(b *BackendBase) Iobuf(
        dim0 []int, 
        x Tensor, 
        dtype base.Dtype, 
        name string,
        persistValues bool, 
        shared Tensor) Tensor {
    if x != nil {
        return x
    }
    var bufshape []int
    if len(dim0) == 2 {
        bufshape = []int{dim0[0], dim0[1]*b.bsz}
    } else {
        bufshape = []int{base.IntsProd(dim0), b.bsz}
    }
    var outTsr Tensor
    if shared != nil {
        if base.IntsEq(shared.Shape(), bufshape) {
            outTsr = shared
        } else {
            // ACHTUNG: Why not to use dtype and name passed to Iobuf?
            outTsr = shared.Share(bufshape, base.DtypeNone, "")
        }
    } else {
        outTsr = b.self.NewTensor(bufshape, dtype)
        outTsr.SetName(name)
        outTsr.SetPersistValues(persistValues)
    }

    outTsr.Assign(b.Int(0))

    return outTsr
}

func(b *BackendBase) SharedIobufSize(shape []int) int {
    return base.IntsProd(shape)
}

func(b *BackendBase) Int(v int) *Int {
    return NewInt(v)
}

func(b *BackendBase) Float(v float64) *Float {
    return NewFloat(v)
}

func(b *BackendBase) Dot(x Value, y Value) Value {
    return BuildOpTreeNode(Dot, x, y)
}

func(b *BackendBase) Add(x Value, y Value) Value {
    return BuildOpTreeNode(Add, x, y)
}

func(b *BackendBase) Subtract(x Value, y Value) Value {
    return BuildOpTreeNode(Sub, x, y)
}

func(b *BackendBase) Multiply(x Value, y Value) Value {
    return BuildOpTreeNode(Mul, x, y)
}

func(b *BackendBase) Divide(x Value, y Value) Value {
    return BuildOpTreeNode(Div, x, y)
}

func(b *BackendBase) TrueDivide(x Value, y Value) Value {
    return BuildOpTreeNode(Div, x, y)
}

func(b *BackendBase) Power(x Value, y Value) Value {
    return BuildOpTreeNode(Pow, x, y)
}

func(b *BackendBase) Reciprocal(x Value) Value {
    return BuildOpTreeNode(Div, b.Float(1.0), x)
}

func(b *BackendBase) Negative(x Value) Value {
    return BuildOpTreeNode(Neg, x, nil)
}

func(b *BackendBase) Sgn(x Value) Value {
    return BuildOpTreeNode(Sgn, x, nil)
}

func(b *BackendBase) Absolute(x Value) Value {
    return BuildOpTreeNode(Abs, x, nil)
}

func(b *BackendBase) Fabs(x Value) Value {
    return BuildOpTreeNode(Abs, x, nil)
}

func(b *BackendBase) Sqrt(x Value) Value {
    return BuildOpTreeNode(Sqrt, x, nil)
}

func(b *BackendBase) Square(x Value) Value {
    return BuildOpTreeNode(Sqr, x, nil)
}

func(b *BackendBase) Exp(x Value) Value {
    return BuildOpTreeNode(Exp, x, nil)
}

func(b *BackendBase) Exp2(x Value) Value {
    return BuildOpTreeNode(Exp2, x, nil)
}

func(b *BackendBase) Safelog(x Value) Value {
    return BuildOpTreeNode(Safelog, x, nil)
}

func(b *BackendBase) Log(x Value) Value {
    return BuildOpTreeNode(Log, x, nil)
}

func(b *BackendBase) Log2(x Value) Value {
    return BuildOpTreeNode(Log2, x, nil)
}

func(b *BackendBase) Sig(x Value) Value {
    return BuildOpTreeNode(Sig, x, nil)
}

func(b *BackendBase) Sig2(x Value) Value {
    return BuildOpTreeNode(Sig2, x, nil)
}

func(b *BackendBase) Tanh(x Value) Value {
    return BuildOpTreeNode(Tanh, x, nil)
}

func(b *BackendBase) Tanh2(x Value) Value {
    return BuildOpTreeNode(Tanh2, x, nil)
}

func(b *BackendBase) Finite(x Value) Value {
    return BuildOpTreeNode(Finite, x, nil)
}

func(b *BackendBase) Rint(x Value) Value {
    return BuildOpTreeNode(Rint, x, nil)
}

/* SKIPPED: Method has different signature in actual backends
func(b *BackendBase) Binarize(x Value, stochastic bool) Value {
    return BuildOpTreeNode(Binarize, x, nil).SetStochastic(stochastic)
}
*/

func(b *BackendBase) Equal(x Value, y Value) Value {
    return BuildOpTreeNode(Eq, x, y)
}

func(b *BackendBase) NotEqual(x Value, y Value) Value {
    return BuildOpTreeNode(Ne, x, y)
}

func(b *BackendBase) Less(x Value, y Value) Value {
    return BuildOpTreeNode(Lt, x, y)
}

func(b *BackendBase) LessEqual(x Value, y Value) Value {
    return BuildOpTreeNode(Le, x, y)
}

func(b *BackendBase) Greater(x Value, y Value) Value {
    return BuildOpTreeNode(Gt, x, y)
}

func(b *BackendBase) GreaterEqual(x Value, y Value) Value {
    return BuildOpTreeNode(Ge, x, y)
}

func(b *BackendBase) Maximum(x Value, y Value) Value {
    return BuildOpTreeNode(Maximum, x, y)
}

func(b *BackendBase) Minimum(x Value, y Value) Value {
    return BuildOpTreeNode(Minimum, x, y)
}

func(b *BackendBase) Shift(x Value, y Value, value bool) Value {
    return BuildOpTreeNode(Shift, x, y).SetValue(value)
}

func(b *BackendBase) Clip(x Value, xMin Value, xMax Value) Value {
    return b.Minimum(b.Maximum(x, xMin), xMax)
}

func(b *BackendBase) Sum(x Value, axis int) Value {
    if axis == base.IntNone {
        t := BuildOpTreeNode(Sum, x, nil).SetAxis(0)
        return BuildOpTreeNode(Sum, t, nil).SetAxis(1)
    }
    return BuildOpTreeNode(Sum, x, nil).SetAxis(axis)
}

func(b *BackendBase) Max(x Value, axis int) Value {
    if axis == base.IntNone {
        t := BuildOpTreeNode(Max, x, nil).SetAxis(0)
        return BuildOpTreeNode(Max, t, nil).SetAxis(1)
    }
    return BuildOpTreeNode(Max, x, nil).SetAxis(axis)
}

func(b *BackendBase) Min(x Value, axis int) Value {
    if axis == base.IntNone {
        t := BuildOpTreeNode(Min, x, nil).SetAxis(0)
        return BuildOpTreeNode(Min, t, nil).SetAxis(1)
    }
    return BuildOpTreeNode(Min, x, nil).SetAxis(axis)
}

func(b *BackendBase) Argmax(x Value, axis int) Value {
    if axis == base.IntNone {
        axis = 1
    }
    return BuildOpTreeNode(Argmax, x, nil).SetAxis(axis)
}

func(b *BackendBase) Argmin(x Value, axis int) Value {
    if axis == base.IntNone {
        axis = 1
    }
    return BuildOpTreeNode(Argmin, x, nil).SetAxis(axis)
}

func(b *BackendBase) Mean(x Value, axis int) Value {
    shape := x.Shape()
    if axis == base.IntNone {
        return b.Multiply(b.Sum(x, base.IntNone), b.Float(1.0/float64(shape[0]*shape[1])))
    }
    return b.Multiply(b.Sum(x, axis), b.Float(1.0/float64(shape[axis])))
}

func(b *BackendBase) Var(x Value, axis int, binary bool) Value {
    // axis may be IntNone
    t := x.Sub(b.Mean(x, axis))
    if binary {
        return b.Mean(b.Shift(t, t, true), axis)
    }
    return b.Mean(b.Square(t), axis)
}

func(b *BackendBase) Std(x Value, axis int) Value {
    return b.Sqrt(b.Var(x, axis, false))
}

func(b *BackendBase) Take(x Tensor, indices Tensor, axis int) Tensor {
    return x.Take(indices, axis)
}

func(b *BackendBase) Onehot(indices Tensor, axis int) Value {
    if axis != 0 && axis != 1 {
        base.ValueError("bad axis for onehot")
    }
    base.AssertMsg(indices.Dtype() == base.Int32 || indices.Dtype() == base.Uint32,
        "should be int32 or uint32, got %s", indices.Dtype().String())
    return BuildOpTreeNode(Onehot, nil, nil).SetIdx(indices).SetAxis(axis)
}

//
//    OpTreeNode
//

type Stack []Value

type OpTreeNode struct {
    ValueBase
    op Op
    a Value
    b Value
    shape []int
    axis int        // reductions, Onehot
    stochastic bool // Binarize
    value bool      // Shift
    idx Tensor      // Onehot
}

// raw constructor used mainly for building stacks
func NewOpTreeNode(op Op, a Value, b Value) *OpTreeNode {
    n := &OpTreeNode{op: op, a: a, b: b}
    n.ValueBase.Init(n)
    return n
}

func(n *OpTreeNode) Init(self Value, op Op, a Value, b Value) {
    n.ValueBase.Init(self)
    n.op = op
    n.a = a
    n.b = b
}

func(n *OpTreeNode) IntrinsicKeyMaps() (string, map[Tensor]int, []Tensor) {
    stack := n.Traverse(nil)
    tensorIndexMap := make(map[Tensor]int)
    var indexTensorMap []Tensor
    var keys []string
    for _, s := range stack {
        switch v := s.(type) {
        case *OpTreeNode:
            op := v.Op()
            if axis := v.Axis(); axis != base.IntNone {
                keys = append(keys, fmt.Sprintf("[%d %d]", op, axis))
            } else {
                keys = append(keys, fmt.Sprintf("%d", op))
            }
        case Tensor:
            // use integer to replace tensor
            index, ok := tensorIndexMap[v]; 
            if !ok {
                // put tensor in dict
                index = len(indexTensorMap)
                tensorIndexMap[v] = index
                indexTensorMap = append(indexTensorMap, v) 
            }
            keys = append(keys, fmt.Sprintf("%d %v", index, v.Shape()))
        }
    }
    return fmt.Sprintf("%v", keys), tensorIndexMap, indexTensorMap
}

func BuildOpTreeNode(op Op, a Value, b Value) *OpTreeNode {
    outShape := []int{1, 1}
    aShape := getOperandShape(a)
    bShape := getOperandShape(b)

    // TODO(orig): Fix shape in smarter way
    aShape = base.IntsExtend(aShape, 2, 1)
    bShape = base.IntsExtend(bShape, 2, 1)

    switch {
    case IsEwOp(op):
        outShape[0] = aShape[0]
        if bShape[0] > aShape[0] {
            outShape[0] = bShape[0]
        }
        outShape[1] = aShape[1]
        if bShape[1] > aShape[1] {
            outShape[1] = bShape[1]
        }
    case IsReductionOp(op):
        // ok: axis is to be set later
    case op == Dot:
        base.Assert(len(aShape) == len(bShape) && len(bShape) == 2 && aShape[1] == bShape[0])
        outShape = []int{aShape[0], bShape[1]}
    case op == Transpose:
        base.Assert(b == nil)
        n := len(aShape)
        for i := 0; i < n; i++ {
            outShape[i] = aShape[n-i-1]
        }
    default:
        base.TypeError("%d is not a valid operation", op)
    }

    n := &OpTreeNode{
        op: op, 
        a: a, 
        b: b,
        shape: outShape, 
        // defaults for respective ops
        axis: base.IntNone,
        stochastic: true,
        value: true,
        idx: nil,
    }
    n.ValueBase.Init(n)
    return n
}

func getOperandShape(x Value) []int {
    if x != nil {
        return x.Shape()
    }
    return []int{0, 0}
}

func(n *OpTreeNode) Op() Op {
    return n.op
}

func(n *OpTreeNode) Left() Value {
    return n.a
}

func(n *OpTreeNode) Right() Value {
    return n.b
}

func(n *OpTreeNode) Shape() []int {
    return n.shape
}

func(n *OpTreeNode) Axis() int {
    return n.axis
}

func(n *OpTreeNode) SetAxis(axis int) *OpTreeNode {
    n.axis = axis
    n.shape[axis] = 1
    return n
}

func(n *OpTreeNode) Stochastic() bool {
    return n.stochastic
}

func(n *OpTreeNode) SetStochastic(stochastic bool) *OpTreeNode {
    n.stochastic = stochastic
    return n
}

func(n *OpTreeNode) Value() bool {
    return n.value
}

func(n *OpTreeNode) SetValue(value bool) *OpTreeNode {
    n.value = value
    return n
}

func(n *OpTreeNode) Idx() Tensor {
    return n.idx
}

func(n *OpTreeNode) SetIdx(idx Tensor) *OpTreeNode {
    n.idx = idx
    return n
}

func(n *OpTreeNode) String() string {
    var left string
    if n.a != nil {
        left = n.a.String()
    } else {
        left = "nil"
    }
    var right string
    if n.b != nil {
        right = n.b.String()
    } else {
        right = "nil"
    }
    return fmt.Sprintf("op(%s, %s, %s)", n.op, left, right)
}

func(n *OpTreeNode) Traverse(stack Stack) Stack {
    // Left
    if a := n.a; a != nil {
        if v, ok := a.(*OpTreeNode); ok {
            stack = v.Traverse(stack)
        } else {
            stack = append(stack, a)
        }
    }

    // Right
    if b := n.b; b != nil {
        if v, ok := b.(*OpTreeNode); ok {
            stack = v.Traverse(stack)
        } else {
            stack = append(stack, b)
        }
    }

    stack = append(stack, n)

    return stack
}

