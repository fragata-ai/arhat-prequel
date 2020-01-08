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

package cuda

var shiftElementCode = `
__device__ static float shift_element(float a, float b, bool value) {
    float result;

    int expb;
    if (value && b == 0) {
        return 0;
    } else if (value) {
        expb = round(log2(abs(b)));
    } else {
        expb = b;
    }

    int expa;
    double mantissa = frexp(a, &expa);
    result = ldexp(mantissa, expa + expb);

    if (value && b < 0) 
        result = -result;

    return result;
}
`
// end shiftElementCode

func shiftElement() string {
    return shiftElementCode
}

