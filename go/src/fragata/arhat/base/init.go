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

package base

import (
    "fmt"
    "math"
    "strings"
)

//
//    Constants
//

var (
    IntNone = -int(^uint(0) >> 1) - 1
    FloatNone = -math.MaxFloat64
)

//
//    Errors
//

func Assert(cond bool) {
    if !cond {
        panic("AssertionError")
    }
}

func AssertMsg(cond bool, msg string, args ...interface{}) {
    if !cond {
        AssertionError(msg, args...)
    }
}

func AssertionError(msg string, args ...interface{}) {
    panic("AssertionError: "+fmt.Sprintf(msg, args...))
}

func TypeError(msg string, args ...interface{}) {
    panic("TypeError: "+fmt.Sprintf(msg, args...))
}

func ValueError(msg string, args ...interface{}) {
    panic("ValueError: "+fmt.Sprintf(msg, args...))
}

func IndexError(msg string, args ...interface{}) {
    panic("IndexError: "+fmt.Sprintf(msg, args...))
}

func AttributeError(msg string, args ...interface{}) {
    panic("AttributeError: "+fmt.Sprintf(msg, args...))
}

func RuntimeError(msg string, args ...interface{}) {
    panic("RuntimeError: "+fmt.Sprintf(msg, args...))
}

func NotImplementedError() {
    panic("NotImplementedError")
}

//
//    Object
//

var counter int

type Object interface {
    Self() Object
    Name() string
    ClassName() string
    ShortClassName() string
}

//
//    ObjectBase
//

type ObjectBase struct {
    self Object
    name string
    desc string
}

var objectBaseInitArgMap = ArgMap{
    "name": NewStringArgOpt(""),
}

func(o *ObjectBase) Init(self Object, args Args) {
    args = objectBaseInitArgMap.Expand(args)
    o.self = self
    name := args["name"].(string)
    if name == "" {
        o.name = fmt.Sprintf("%s_%d", o.ShortClassName(), counter)
    } else {
        o.name = name
    }
    counter++
}

func(o *ObjectBase) Self() Object {
    return o.self
}

func(o *ObjectBase) Name() string {
    return o.name
}

func(o *ObjectBase) ShortClassName() string {
    name := o.self.ClassName()
    if pos := strings.LastIndex(name, "."); pos >= 0 {
        name = name[pos+1:]
    }
    return name
}

//
//    Simple helpers
//

func IntMax(x int, y int) int {
    if x > y {
        return x
    } else {
        return y
    }
}

func IntMin(x int, y int) int {
    if x < y {
        return x
    } else {
        return y
    }
}

func ResolveInt(value int, defval int) int {
    if value == IntNone {
        value = defval
    }
    return value
}

func ResolveFloat(value float64, defval float64) float64 {
    if value == FloatNone {
        value = defval
    }
    return value
}

func ResolveDtype(value Dtype, defval Dtype) Dtype {
    if value == DtypeNone {
        value = defval
    }
    return value
}

//
//    Slices
//

func IntsMax(a []int) int {
    n := len(a)
    y := a[0]
    for i := 1; i < n; i++ {
        if x := a[i]; x > y {
            y = x
        }
    }
    return y
}

func IntsMin(a []int) int {
    n := len(a)
    y := a[0]
    for i := 1; i < n; i++ {
        if x := a[i]; x < y {
            y = x
        }
    }
    return y
}

func IntsProd(a []int) int {
    p := 1
    for _, v := range a {
        p *= v
    }
    return p
}

func IntsFill(a []int, v int) {
    n := len(a)
    for i := 0; i < n; i++ {
        a[i] = v
    }
}

func IntsFind(a []int, v int) int {
    for i, x := range a {
        if x == v {
            return i
        }
    }
    return -1
}

func IntsCopy(a []int) []int {
    b := make([]int, len(a))
    copy(b, a)
    return b
}

func IntsReverse(a []int) []int {
    n := len(a)
    b := make([]int, n)
    for i := 0; i < n; i++ {
        b[i] = a[n-1-i]
    }
    return b
}

func IntsExtend(a []int, n int, v int) []int {
    k := len(a)
    if k >= n {
        return a
    }
    b := make([]int, n)
    copy(b, a)
    if v != 0 {
        IntsFill(b[k:], v)
    }
    return b
}

func IntsEq(a []int, b []int) bool {
    n := len(a)
    if len(b) != n {
        return false
    }
    for i := 0; i < n; i++ {
        if a[i] != b[i] {
            return false;
        }
    }
    return true
}

