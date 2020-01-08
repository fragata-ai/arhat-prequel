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

package generators

import (
    "fmt"
    "fragata/arhat/base"
    "strings"
)

//
//    Code
//

type Code struct {
    builder strings.Builder
    indent int
}

func NewCode() *Code {
    return new(Code)
}

func(c *Code) Reset() {
    c.builder.Reset()
    c.indent = 0
}

func(c *Code) WriteLine(s string, args ...interface{}) {
    line := strings.Repeat(" ", 4*c.indent) + fmt.Sprintf(s, args...) + "\n"
    _, err := c.builder.WriteString(line)
    base.Assert(err == nil)
}

func(c *Code) WriteChunk(chunk string) {
    _, err := c.builder.WriteString(chunk)
    base.Assert(err == nil)    
}

func(c *Code) String() string {
    return c.builder.String()
}

func(c *Code) Indent(delta int) {
    c.indent += delta
    if c.indent < 0 {
        c.indent = 0
    }
}

