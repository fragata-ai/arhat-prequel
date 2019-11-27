//
// Copyright 2019 FRAGATA COMPUTER SYSTEMS AG
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

package generators

import (
    "fmt"
    "fragata/arhat/backends"
    "fragata/arhat/base"
    "strings"
)

//
//    IntSymbol
//

type IntSymbol struct {
    backends.Int
    symbol string
}

func NewIntSymbol(s string) *IntSymbol {
    x := new(IntSymbol)
    x.Init(x, s)
    return x
}

func(x *IntSymbol) Init(self backends.Value, s string) {
    x.Int.Init(self, 0)
    x.symbol = s
}

func(x *IntSymbol) Symbol() string {
    return x.symbol
}

func(x *IntSymbol) String() string {
    return x.symbol
}

//
//    FloatSymbol
//

type FloatSymbol struct {
    backends.Float
    symbol string
}

func NewFloatSymbol(s string) *FloatSymbol {
    x := new(FloatSymbol)
    x.Init(x, s)
    return x
}

func(x *FloatSymbol) Init(self backends.Value, s string) {
    x.Float.Init(self, 0.0)
    x.symbol = s
}

func(x *FloatSymbol) Symbol() string {
    return x.symbol
}

func(x *FloatSymbol) String() string {
    return x.symbol
}

//
//    GlobalObject
//

type GlobalObject interface {
    Declare()
    Initialize()
}

//
//    Generator
//

type Generator interface {
    backends.Backend
    HostChunks() []string
    DeviceChunks() []string
    StartCode()
    EndCode()
    PushCode()
    PopCode()
    GetCode() *Code
    WriteLine(s string, args ...interface{})
    WriteChunk(chunk string)
    Indent(delta int)
    EnterInit()
    ExitInit()
    AddGlobalObject(obj GlobalObject)
    BuildProlog()
    MakeIndex(name string) int
    MakeIntSymbol() *IntSymbol
    MakeFloatSymbol() *FloatSymbol
    // following methods must be overriden
    GetData(dest string, start string, stop string, x backends.Tensor) string
    GetMetricSum(x backends.Tensor, start string, stop string) string
    OutputCode(outDir string) error
    FormatBufferRef(tensor backends.Tensor, paren bool) string
}

//
//    GeneratorBase
//

type writeMode int

const (
    writeModeHost writeMode = iota
    writeModeInit
    writeModeProlog
    writeModeDevice
)

type GeneratorBase struct {
    backends.BackendBase
    hostChunks []string
    initChunks []string
    prologChunks []string
    deviceChunks []string
    codeStack []*Code
    codeStackTop int
    writeMode writeMode
    nextSymbolIndex int
    nextIndex map[string]int
    globalObjects []GlobalObject
}

func(b *GeneratorBase) Init(
        self Generator, 
        rngSeed int, 
        defaultDtype base.Dtype, 
        compatMode backends.CompatMode) {
    b.BackendBase.Init(self, rngSeed, defaultDtype, compatMode)
    b.codeStack = make([]*Code, 16)
    for i := 0; i < 16; i++ {
        b.codeStack[i] = NewCode()
    }
    b.codeStackTop = 0
    b.writeMode = writeModeHost
    b.nextSymbolIndex = 1
    b.nextIndex = make(map[string]int)
}

func(b *GeneratorBase) HostChunks() []string {
    return b.hostChunks
}

func(b *GeneratorBase) DeviceChunks() []string {
    return b.deviceChunks
}

func(b *GeneratorBase) GetLearningRate(
        schedule backends.Schedule, learningRate float64, epoch backends.Value) backends.Value {
    epochSymbol := epoch.(*IntSymbol).Symbol()
    result := b.MakeFloatSymbol()
    b.WriteLine("float %s = schedule->GetLearningRate(%s, %s);", 
        result.Symbol(), FormatFloat32(learningRate), epochSymbol)
    return result
}

func(b *GeneratorBase) StartCode() {
    b.codeStackTop = 0
    b.writeMode = writeModeHost
}

func(b *GeneratorBase) EndCode() {
    base.Assert(b.codeStackTop == 0)
    b.Self().(Generator).BuildProlog()
    chunk := b.codeStack[0].String()
    if len(chunk) != 0 {
        b.hostChunks = append(b.hostChunks, chunk)
    }
    b.insertInitChunks()
    b.codeStack[0].Reset()
}

func(b *GeneratorBase) insertInitChunks() {
    n1 := len(b.prologChunks)
    n2 := len(b.initChunks)
    n3 := len(b.hostChunks)
    chunks := make([]string, n1+n2+n3)
    pos := 0
    if n1 != 0 {
        copy(chunks, b.prologChunks)
        pos += n1
    }
    if n2 != 0 {
        copy(chunks[pos:], b.initChunks)
        pos += n2
    }
    if n3 != 0 {
        copy(chunks[pos:], b.hostChunks)
    }
    b.hostChunks = chunks
}

func(b *GeneratorBase) PushCode() {
    next := b.codeStackTop + 1
    if next >= len(b.codeStack) {
        b.codeStack = append(b.codeStack, NewCode())
    }
    b.codeStackTop = next
}

func(b *GeneratorBase) PopCode() {
    prev := b.codeStackTop - 1
    base.Assert(prev >= 0)
    code := b.codeStack[b.codeStackTop]
    chunk := code.String()
    code.Reset()
    if len(chunk) != 0 {
        switch b.writeMode {
        case writeModeHost:
            b.hostChunks = append(b.hostChunks, chunk)
        case writeModeInit:
            b.initChunks = append(b.initChunks, chunk)
        case writeModeProlog:
            b.prologChunks = append(b.prologChunks, chunk)
        case writeModeDevice:
            b.deviceChunks = append(b.deviceChunks, chunk)
        }
    }
    b.codeStackTop = prev
}

func(b *GeneratorBase) GetCode() *Code {
    return b.codeStack[b.codeStackTop]
}

func(b *GeneratorBase) WriteLine(s string, args ...interface{}) {
    b.codeStack[b.codeStackTop].WriteLine(s, args...)
}

func(b *GeneratorBase) WriteChunk(chunk string) {
    b.codeStack[b.codeStackTop].WriteChunk(chunk)
}

func(b *GeneratorBase) Indent(delta int) {
    b.codeStack[b.codeStackTop].Indent(delta)
}

func(b *GeneratorBase) EnterInit() {
    b.enterMode(writeModeInit)
}

func(b *GeneratorBase) ExitInit() {
    b.exitMode(writeModeInit)
}

func(b *GeneratorBase) EnterProlog() {
    b.enterMode(writeModeProlog)
}

func(b *GeneratorBase) ExitProlog() {
    b.exitMode(writeModeProlog)
}

func(b *GeneratorBase) EnterKernel() {
    b.enterMode(writeModeDevice)
}

func(b *GeneratorBase) ExitKernel() {
    b.exitMode(writeModeDevice)
}

func(b *GeneratorBase) enterMode(mode writeMode) {
    base.Assert(b.writeMode == writeModeHost)
    b.writeMode = mode
    b.PushCode()
}

func(b *GeneratorBase) exitMode(mode writeMode) {
    base.Assert(b.writeMode == mode)
    b.PopCode()
    b.writeMode = writeModeHost
}

func(b *GeneratorBase) AddGlobalObject(obj GlobalObject) {
    b.globalObjects = append(b.globalObjects, obj)
}

func(b *GeneratorBase) DeclareGlobalObjects() {
    if len(b.globalObjects) == 0 {
        return
    }
    for _, obj := range b.globalObjects {
        obj.Declare()
    }
    b.WriteLine("")
}

func(b *GeneratorBase) InitializeGlobalObjects() {
    for _, obj := range b.globalObjects {
        obj.Initialize()
    }
}

func(b *GeneratorBase) MakeIndex(name string) int {
    index, ok := b.nextIndex[name]
    if !ok {
        index = 1
    }
    b.nextIndex[name] = index + 1
    return index
}

func(b *GeneratorBase) MakeIntSymbol() *IntSymbol {
    return NewIntSymbol(b.makeSymbolName())
}

func(b *GeneratorBase) MakeFloatSymbol() *FloatSymbol {
    return NewFloatSymbol(b.makeSymbolName())
}

func(b *GeneratorBase) makeSymbolName() string {
    id := b.nextSymbolIndex
    b.nextSymbolIndex++
    return fmt.Sprintf("tmp_%d", id)
}

// public helper functions

var scalarOps = map[backends.Op]string{
    backends.Add: "(%s + %s)",
    backends.Sub: "(%s - %s)",
    backends.Mul: "(%s * %s)",
    backends.Div: "(%s / %s)",
    backends.Eq: "(%s == %s)",
    backends.Ne: "(%s != %s)",
    backends.Lt: "(%s < %s)",
    backends.Le: "(%s <= %s)",
    backends.Gt: "(%s > %s)",
    backends.Ge: "(%s >= %s)",
    backends.Minimum: "fminf(%s, %s)",
    backends.Maximum: "fmaxf(%s, %s)",
    backends.Pow: "powf(%s, %s)",
    backends.Finite: "IsFinite(%s)",
    backends.Neg: "(-%s)",
    backends.Abs: "abs(%s)",
    backends.Sgn: "Sgnf(%s)",
    backends.Sqrt: "sqrtf(%s)",
    backends.Sqr: "Sqrf(%s)",
    backends.Exp: "expf(%s)",
    backends.Log: "logf(%s)",
    backends.Safelog: "Safelogf(%s)",
    backends.Exp2: "exp2f(%s)",
    backends.Log2: "log2f(%s)",
    backends.Sig: "Sigf(%s)",
    backends.Sig2: "Sig2f(%s)",
    backends.Tanh: "tanhf(%s)",
    backends.Tanh2: "Tanh2f(%s)",
    // SKIPPED: backends.Rand
    // SKIPPED: backends.Onehot
}

func FormatScalar(x backends.Value) string {
    switch v := x.(type) {
    case *IntSymbol:
        return v.symbol
    case *FloatSymbol:
        return v.symbol
    case *backends.Int:
        return fmt.Sprintf("%d", v.Value())
    case *backends.Float:
        // always represented as float in C++ code
        return FormatFloat32(v.Value())
    case *backends.OpTreeNode:
        op := v.Op()
        tmpl, ok := scalarOps[op]
        if !ok {
            base.ValueError("Unsupported scalar op: %s", op)
            return "0"
        }
        left := v.Left()
        right := v.Right()
        if backends.IsUnaryOp(op) {
            l := FormatScalar(left)
            return fmt.Sprintf(tmpl, l)
        }
        if backends.IsBinaryOp(op) {
            l := FormatScalar(left)
            r := FormatScalar(right)
            return fmt.Sprintf(tmpl, l, r)
        }
        base.Assert(false)
        return "0"
    default:
        base.ValueError("Invalid scalar node type: %t", x)
        return "0"
    }
}

func FormatFloat32(x float64) string {
    s := fmt.Sprintf("%.7g", x)
    if !strings.Contains(s, ".") && !strings.Contains(s, "e") {
        s += ".0"
    }
    s += "f"
    return s
}

