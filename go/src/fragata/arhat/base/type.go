//
// Copyright 2019 FRAGATA COMPUTER SYSTEMS AG
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

//
//    Data types
//

type Dtype int

const (
    DtypeNone Dtype = iota
    Float16
    Float32
    Float64
    Int8
    Uint8
    Int16
    Uint16
    Int32
    Uint32
    Int64
    Uint64
    numDtypes
)

const (
    isInt = 1 << iota
    isUint
    isFloat
)

type dtypeProp struct {
    str string
    kind int
    itemSize int
}

var dtypePropMap = [numDtypes]dtypeProp {
    DtypeNone: {"None", 0, 0},
    Float16:   {"float16", isFloat, 2},
    Float32:   {"float32", isFloat, 4},
    Float64:   {"float64", isFloat, 8},
    Int8:      {"int8", isInt, 1},
    Uint8:     {"uint8", isUint, 1},
    Int16:     {"int16", isInt, 2},
    Uint16:    {"uint16", isUint, 2},
    Int32:     {"int32", isInt, 4},
    Uint32:    {"uint32", isUint, 4},
    Int64:     {"int64", isInt, 8},
    Uint64:    {"uint64", isUint, 8},
}

func(t Dtype) String() string {
    return dtypePropMap[t].str
}

func(t Dtype) IsInt() bool {
    return (dtypePropMap[t].kind & isInt != 0)
}

func(t Dtype) IsUint() bool {
    return (dtypePropMap[t].kind & isUint != 0)
}

func(t Dtype) IsFloat() bool {
    return (dtypePropMap[t].kind & isFloat != 0)
}

func(t Dtype) ItemSize() int {
    return dtypePropMap[t].itemSize
}

