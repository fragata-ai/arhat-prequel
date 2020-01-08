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

package base

import "fmt"

//
//    Keyword arguments
//

type Args map[string]interface{}

var ArgsNone = Args{}

func MakeArgs(args []interface{}) Args {
    n := len(args)
    if n == 1 {
        // passthru Args, for internal use
        if a, ok := args[0].(Args); ok {
            return a
        }
    }
    out := make(Args)
    expandArgs(args, out)
    return out
}

func expandArgs(args []interface{}, out Args) {
    n := len(args)
    if n % 2 != 0 {
        ValueError("number of dynamic arguments must be even, got %d", n)
    }
    for i := 0; i < n; i += 2 {
        key, ok := args[i].(string)
        if !ok {
            ValueError("dynamic argument #%d: need string key, got %v", i/2+1, args[i])
        }
        value := args[i+1]
        if key == "*" {
            nestedArgs, ok := value.([]interface{})
            if !ok {
                ValueError("dynamic argument #%d: need slice value, got %v", i/2+1, args[i])
            }
            expandArgs(nestedArgs, out)
        } else {
            if _, ok := out[key]; ok {
                ValueError("dynamic argument #%d: duplicate key %s", i/2+1, key)
            }
            out[key] = value
        }
    }
}

func(a Args) Filter(keys []string) Args {
    result := make(Args)
    for _, key := range keys {
        value, ok := a[key]
        if  ok {
            result[key] = value
        }
    }
    return result
}

type ArgDef interface {
    Required() bool
    Default() interface{}
    Expand(v interface{}) (interface{}, bool)
}

type ArgDefBase struct {
    required bool
    defval interface{}
}

func(a *ArgDefBase) Init(required bool, defval interface{}) {
    a.required = required
    a.defval = defval
}

func(a *ArgDefBase) Required() bool {
    return a.required
}

func(a *ArgDefBase) Default() interface{} {
    return a.defval
}

type AnyArg struct {
    ArgDefBase
}

func NewAnyArg() *AnyArg {
    return &AnyArg{ArgDefBase{true, nil}}
}

func NewAnyArgOpt(defval interface{}) *AnyArg {
    return &AnyArg{ArgDefBase{false, defval}}
}

func(a *AnyArg) Expand(v interface{}) (interface{}, bool) {
    return v, true
}

type BoolArg struct {
    ArgDefBase
}

func NewBoolArg() *BoolArg {
    return &BoolArg{ArgDefBase{true, nil}}
}

func NewBoolArgOpt(defval bool) *BoolArg {
    return &BoolArg{ArgDefBase{false, defval}}
}

func(a *BoolArg) Expand(v interface{}) (interface{}, bool) {
    result, ok := v.(bool)
    if !ok {
        return nil, false
    }
    return result, true
}

type IntArg struct {
    ArgDefBase
}

func NewIntArg() *IntArg {
    return &IntArg{ArgDefBase{true, nil}}
}

func NewIntArgOpt(defval int) *IntArg {
    return &IntArg{ArgDefBase{false, defval}}
}

func(a *IntArg) Expand(v interface{}) (interface{}, bool) {
    _, ok := v.(int)
    if !ok {
        return nil, false
    }
    return v, true
}

type FloatArg struct {
    ArgDefBase
}

func NewFloatArg() *FloatArg {
    return &FloatArg{ArgDefBase{true, nil}}
}

func NewFloatArgOpt(defval float64) *FloatArg {
    return &FloatArg{ArgDefBase{false, defval}}
}

func(a *FloatArg) Expand(v interface{}) (interface{}, bool) {
    _, ok := v.(float64)
    if !ok {
        return nil, false
    }
    return v, true
}

type StringArg struct {
    ArgDefBase
}

func NewStringArg() *StringArg {
    return &StringArg{ArgDefBase{true, nil}}
}

func NewStringArgOpt(defval string) *StringArg {
    return &StringArg{ArgDefBase{false, defval}}
}

func(a *StringArg) Expand(v interface{}) (interface{}, bool) {
    _, ok := v.(string)
    if !ok {
        return nil, false
    }
    return v, true
}

type EnumDef map[string]int

type EnumArg struct {
    ArgDefBase
    enum EnumDef
}

func NewEnumArg(enum EnumDef) *EnumArg {
    return &EnumArg{ArgDefBase{true, nil}, enum}
}

func NewEnumArgOpt(enum EnumDef, defval string) *EnumArg {
    return &EnumArg{ArgDefBase{false, defval}, enum}
}

func(a *EnumArg) Expand(v interface{}) (interface{}, bool) {
    switch t := v.(type) {
    case int:
        // already translated
        return v, true
    case string:
        result, ok := a.enum[t]
        if !ok {
            return nil, false
        }
        return result, true
    default:
        return nil, false
    }
}

type IntListArg struct {
    ArgDefBase
}

func NewIntListArg() *IntListArg {
    return &IntListArg{ArgDefBase{true, nil}}
}

func NewIntListArgOpt(defval []int) *IntListArg {
    return &IntListArg{ArgDefBase{false, defval}}
}

func(a *IntListArg) Expand(v interface{}) (interface{}, bool) {
    switch t := v.(type) {
    case int:
        return []int{t}, true
    case []int:
        return v, true
    default:
        return nil, false
    }
}

func ToIntList(v interface{}) []int {
    if v == nil {
        return nil
    }
    return v.([]int)
}

type FloatListArg struct {
    ArgDefBase
}

func NewFloatListArg() *FloatListArg {
    return &FloatListArg{ArgDefBase{true, nil}}
}

func NewFloatListArgOpt(defval []float64) *FloatListArg {
    return &FloatListArg{ArgDefBase{false, defval}}
}

func(a *FloatListArg) Expand(v interface{}) (interface{}, bool) {
    switch t := v.(type) {
    case float64:
        return []float64{t}, true
    case []float64:
        return v, true
    default:
        return nil, false
    }
}

func ToFloatList(v interface{}) []float64 {
    if v == nil {
        return nil
    }
    return v.([]float64)
}

//
//    ArgMap
//

type ArgMap map[string]ArgDef

func(m ArgMap) Expand(args Args) Args {
    if args == nil {
        args = ArgsNone
    }
    for key, _ := range args {
        _, ok := m[key]
        if !ok {
            InvalidArgument(key)
        }
    }
    result := make(Args)
    for key, def := range m {
        value, ok := args[key]
        if !ok {
            if def.Required() {
                MissingArgument(key)
            }
            value = def.Default()
        }
        value, ok = def.Expand(value)
        if !ok {
            InvalidArgument(key)
        }
        result[key] = value
    }
    return result
}

func InvalidArgument(name string) {
    panic(fmt.Sprintf("Invalid argument: %s", name))
}

func MissingArgument(name string) {
    panic(fmt.Sprintf("Missing argument: %s", name))
}

func ArgumentError(msg string) {
    panic(fmt.Sprintf("Argument error: %s", msg))
}

