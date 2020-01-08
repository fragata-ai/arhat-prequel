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

//
// Based on go-float16 - IEEE 754 binary16 half precision format
// Written in 2013 by h2so5 <mail@h2so5.net>
//
// For details, see https://github.com/h2so5/half/blob/master/float16.go
// The original go-float16 package has been attributed as follows.
//
// To the extent possible under law, the author(s) have dedicated all copyright and
// related and neighboring rights to this software to the public domain worldwide.
// This software is distributed without any warranty.
// You should have received a copy of the CC0 Public Domain Dedication along with this software.
// If not, see <http://creativecommons.org/publicdomain/zero/1.0/>.
//

package base

import "math"

// A Half represents a 16-bit floating point number.
type Half uint16

// NewHalf allocates and returns a new Half set to f.
func NewHalf(f float32) Half {
    i := math.Float32bits(f)
    sign := uint16((i >> 31) & 0x1)
    exp := (i >> 23) & 0xff
    exp16 := int16(exp) - 127 + 15
    frac := uint16(i>>13) & 0x3ff
    if exp == 0 {
        exp16 = 0
    } else if exp == 0xff {
        exp16 = 0x1f
    } else {
        if exp16 > 0x1e {
            exp16 = 0x1f
            frac = 0
        } else if exp16 < 0x01 {
            exp16 = 0
            frac = 0
        }
    }
    f16 := (sign << 15) | uint16(exp16<<10) | frac
    return Half(f16)
}

// Float32 returns the float32 representation of f.
func(f Half) Float32() float32 {
    sign := uint32((f >> 15) & 0x1)
    exp := (f >> 10) & 0x1f
    exp32 := uint32(exp) + 127 - 15
    if exp == 0 {
        exp32 = 0
    } else if exp == 0x1f {
        exp32 = 0xff
    }
    frac := uint32(f & 0x3ff)
    i := (sign << 31) | (exp32 << 23) | (frac << 13)
    return math.Float32frombits(i)
}

