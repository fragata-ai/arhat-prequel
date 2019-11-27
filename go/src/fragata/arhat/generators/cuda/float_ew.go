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

package cuda

import (
    "fmt"
    "fragata/arhat/backends"
    "fragata/arhat/base"
    "fragata/arhat/generators/acc"
    "strings"
    "text/template"
)

//
//    Platform-specific part of kernel builder pass 2
//

func BuildCompoundKernel(typeArgs []acc.TypeArg, computeCapability [2]int) (string, string) {
    // split stack into stages
    stages := acc.BuildStages(typeArgs)

    var stack []string
    stackPop := func() string {
        n := len(stack)
        s := stack[n-1]
        stack = stack[:n-1]
        return s
    }
    var placeholders []string
    stageOutReg := make(map[acc.TypeArg]string)
    argDict := make(map[string]string)
    arrayIds := make(map[int]bool)
    arrayStageIds := make(map[string]bool)
    fp16In := false
    randInit := false
    randFunc := false
    threads := typeArgs[len(typeArgs)-1].(*acc.OpTypeArg).Threads()
    templateSource := ewTemplate
    templateVals := map[string][]string{
        "name": getKernelName(),
        "common": []string{},
        "inits": []string{},
        "finish": []string{},
    }

    for stage, _ := range stages {
        stageType := stages[stage].Kind()
        stageStack := stages[stage].Stack()
        var newPlaceholders []string

        // build out the template as we process stages
        switch stageType {
        case "reduction":
            newPlaceholders = append(newPlaceholders, fmt.Sprintf("loads%d", stage))
            newPlaceholders = append(newPlaceholders, fmt.Sprintf("ops%d", stage))
            newPlaceholders = append(newPlaceholders, fmt.Sprintf("shfl_red%d", stage))
            templateSource += format(stageTemplate["loop"], stage)
            if threads > 32 {
                newPlaceholders = append(newPlaceholders, fmt.Sprintf("var_red%d", stage))
                newPlaceholders = append(newPlaceholders, fmt.Sprintf("share1_red%d", stage))
                newPlaceholders = append(newPlaceholders, fmt.Sprintf("share2_red%d", stage))
                templateSource += format(stageTemplate["red"], stage)
            } else {
                templateSource += format(stageTemplate["red32"], stage)
            }

        case "scalar":
            newPlaceholders = append(newPlaceholders, fmt.Sprintf("ops%d", stage))
            templateSource += format(stageTemplate["red_ops"], stage)

        case "red_out":
            newPlaceholders = append(newPlaceholders, fmt.Sprintf("ops%d", stage))
            templateSource += format(stageTemplate["red_out"], stage)

        default: // "ew_out"
            newPlaceholders = append(newPlaceholders, fmt.Sprintf("loads%d", stage))
            newPlaceholders = append(newPlaceholders, fmt.Sprintf("ops%d", stage))
            templateSource += format(stageTemplate["loop"], stage)
        }

        for _, key := range newPlaceholders {
            templateVals[key] = []string{}
            placeholders = append(placeholders, key)
        }

        outDtype := base.DtypeNone
        outTake := base.IntNone

        for iarg, arg := range stageStack {
            switch v := arg.(type) {
            case *acc.TensorTypeArg:
                // Array operands
                argId := v.Indx()
                dtype := v.Dtype()
                takeAxis := v.TakeAxis()

                idStageKey := fmt.Sprintf("%d %d", argId, stage)

                isOutTensor := (stage == len(stages) - 1 && iarg == 0)

                // first arg is output array, don't put on stack 
                if isOutTensor {
                    outDtype = dtype
                    outTake = takeAxis
                } else {
                    stack = append(stack, fmt.Sprintf("a%d", argId))
                }

                ewDtype := ewTypes[dtype]
                val := []interface{}{argId, stage, ewDtype["type"], ewDtype["cvt"]}

                // First time we see a tensor initialize everything
                if !arrayIds[argId] {
                    arrayIds[argId] = true
                    arrayStageIds[idStageKey] = true

                    var arguments string
                    if isOutTensor {
                        // output tensor
                        ewOut := ewStrings[fmt.Sprintf("out%d", takeAxis)]
                        arguments = format(ewOut["arguments"], val...)
                        templateVals["inits"] = 
                            append(templateVals["inits"], format(ewOut["inits"], val...))
                    } else {
                        // input tensors
                        ewIn := ewStrings[fmt.Sprintf("in%d", takeAxis)]
                        loads := fmt.Sprintf("loads%d", stage)
                        arguments = format(ewIn["arguments"], val...)
                        templateVals["inits"] =
                            append(templateVals["inits"], format(ewIn["inits"], val...))
                        templateVals[loads] =
                            append(templateVals[loads], format(ewIn["loads"], val...))
                    }

                    if dtype == base.Float16 && !fp16In {
                        templateVals["common"] = append(templateVals["common"], commonFp16toFp32)
                        fp16In = true
                    }

                    argDict[arg.Key()] = arguments

                } else if !arrayStageIds[idStageKey] {
                    // Subsequent times we see a tensor just initialize inits and loads
                    arrayStageIds[idStageKey] = true
                    ewIn := ewStrings[fmt.Sprintf("in%d", takeAxis)]
                    loads := fmt.Sprintf("loads%d", stage)
                    templateVals["inits"] = 
                        append(templateVals["inits"], format(ewIn["inits"], val...))
                    templateVals[loads] = 
                        append(templateVals[loads], format(ewIn["loads"], val...))
                }

            case *acc.FloatTypeArg:
                // Constant operands
                argId := v.Indx()
                stack = append(stack, fmt.Sprintf("c%d", argId))
                argKey := arg.Key()
                if _, ok := argDict[argKey]; !ok {
                    argDict[argKey] = format(ewStrings["const"]["arguments"], argId)
                }

            case *acc.SymbolTypeArg:
                // Symbol operands
                argId := v.Indx()
                stack = append(stack, fmt.Sprintf("c%d", argId))
                argKey := arg.Key()
                if _, ok := argDict[argKey]; !ok {
                    argDict[argKey] = format(ewStrings["const"]["arguments"], argId)
                }

            case *acc.OpTypeArg:
                // Operations
                argType := v.OpName()
                argId := v.OpCnt()

                if argType == backends.Assign {
                    ops := fmt.Sprintf("ops%d", stage)

                    // loop end condition for last stage
                    arguments := []string{fmt.Sprintf("const int n%d", stage)}

                    // rounding mode
                    var mode string
                    if v.Rounding() {
                        mode = "random"
                        arguments = append(arguments, "const int mantissa_bits")
                        if !randInit {
                            randInit = initRand(templateVals)
                        }
                        templateVals["inits"] = append(templateVals["inits"], initRandRoundFunc)
                    } else {
                        mode = "nearest"
                    }

                    argDict[arg.Key()] = strings.Join(arguments, ", ")

                    outVal := stackPop()
                    // if the last stack value came from an argmax/min just do
                    // implicit type conversion
                    var roundVal string
                    // ACHTUNG: Where does 'i' come from?
                    if outVal[0] == 'i' && (outDtype.IsInt() || outDtype.IsUint()) {
                        roundVal = outVal
                    } else {
                        if ewCommon, ok := commonRound[mode][outDtype]; ok {
                            templateVals["common"] = append(templateVals["common"], ewCommon)
                        }
                        if ewRound, ok := ewStringsRound[mode][outDtype]; ok {
                            roundVal = fmt.Sprintf("r%d", argId)
                            templateVals[ops] = 
                                append(templateVals[ops], format(ewRound, roundVal, outVal))
                        } else {
                            roundVal = outVal
                        }
                    }

                    templateVals[ops] = append(templateVals[ops],
                        format(ewStrings[fmt.Sprintf("out%d", outTake)]["output"], roundVal))

                } else if s, ok := stageOutReg[arg]; ok {
                    stack = append(stack, s)

                } else if templ := getFloatOp(argType); templ != nil {

                    if len(templateVals["name"]) < 16 {
                        templateVals["name"] = append(templateVals["name"], argType.String())
                    }

                    ops := fmt.Sprintf("ops%d", stage)

                    numOps := templ.numOps
                    opCode := templ.opCode

                    if argType == backends.Rand {
                        if !randInit {
                            randInit = initRand(templateVals)
                        }
                        if !randFunc {
                            templateVals["common"] = append(templateVals["common"], commonFrand)
                            randFunc = true
                        }
                    }

                    outReg := fmt.Sprintf("r%d", argId)
                    opList := []interface{}{outReg}

                    // build the operands from the stack
                    for i := 0; i < numOps; i++ {
                        opList = append(opList, stackPop())
                    }

                    if argType == backends.Onehot {
                        hotAxis := v.HotAxis()
                        var testVal string
                        // ACHTUNG: Verify this
                        if hotAxis != 0  {
                            testVal = "i"
                        } else {
                            testVal = "bid"
                        }

                        ewIn := ewStrings[fmt.Sprintf("onehot%d", hotAxis)]
                        loads := fmt.Sprintf("loads%d", stage)
                        templateVals["inits"] = 
                            append(templateVals["inits"], format(ewIn["inits"], argId))
                        templateVals[loads] = 
                            append(templateVals[loads], format(ewIn["loads"], argId))
                        opList = append(opList, fmt.Sprintf("onehot%d", argId))
                        opList = append(opList, testVal)

                        argDict[arg.Key()] = format(ewIn["arguments"], argId)
                    }

                    templateVals[ops] = append(templateVals[ops], format(opCode, opList...))

                    if iarg == len(stageStack) - 1 {
                        // if this is the last op on the current stack, store its register stage
                        // in the stage output dict
                        stageOutReg[arg] = outReg
                    } else {
                        // otherwise push the reg onto the stack as normal
                        stack = append(stack, outReg)
                    }

                } else if templ := getReductionOp(argType); templ != nil {

                    if len(templateVals["name"]) < 16 {
                        templateVals["name"] = append(templateVals["name"], argType.String())
                    }

                    // loop end condition for current stage
                    // add regardless of duplicate reduction stage
                    argDict[arg.Key()] = fmt.Sprintf("const int n%d", stage)

                    // avoid float conversion for argmax/min
                    var reg string
                    if argType == backends.Argmax || argType == backends.Argmin {
                        reg = "i"
                    } else {
                        reg = "r"
                    }

                    ops := fmt.Sprintf("ops%d", stage)
                    shflRed := fmt.Sprintf("shfl_red%d", stage)
                    redArg := fmt.Sprintf("%s%d", reg, argId)
                    redStrings := templ
                    stackArg := stackPop()

                    templateVals["inits"] = 
                        append(templateVals["inits"], format(redStrings["inits"], redArg))
                    templateVals[ops] = 
                        append(templateVals[ops], format(redStrings["ops"], redArg, stackArg))
                    templateVals[shflRed] = 
                        append(templateVals[shflRed], format(redStrings["shfl_red"], redArg))
                    if threads > 32 {
                        varRed := fmt.Sprintf("var_red%d", stage)
                        shr1Red := fmt.Sprintf("share1_red%d", stage)
                        shr2Red := fmt.Sprintf("share2_red%d", stage)
                        templateVals[varRed] = append(templateVals[varRed], redArg)
                        templateVals[shr1Red] = 
                            append(templateVals[shr1Red], format(redStrings["share1_red"], redArg))
                        templateVals[shr2Red] =
                            append(templateVals[shr2Red], format(redStrings["share2_red"], redArg))
                    }

                    // reduction ops are always the last on the stack
                    // just store the register state in the stage output dict
                    stageOutReg[arg] = redArg

                } else {
                    base.ValueError("Bad op type.")
                }
            }
        }
    }

    if (computeCapability[0] == 3 && computeCapability[1] < 5) || computeCapability[0] < 3 {
        templateVals["common"] = append(templateVals["common"], commonKepler)
    }

    templateSource += finTemplate

    // since we reorderd the operations we need to generate the argument list
    // in the original order
    var arguments []string
    unused := 1
    for _, arg := range typeArgs {
        argKey := arg.Key()
        if params, ok := argDict[argKey]; ok {
            arguments = append(arguments, params)
            delete(argDict, argKey)
        } else if v, ok := arg.(*acc.OpTypeArg); ok && getReductionOp(v.OpName()) != nil {
            // fill in the loop counter for the duplicate reductions that were removed
            arguments = append(arguments, fmt.Sprintf("const int unused%d", unused))
            unused++
        }
    }

    name := strings.Join(templateVals["name"], "_")

    // convert lists to strings
    templateParams := map[string]interface{}{
        "threads": threads,
        "name": name,
        "common": strings.Join(templateVals["common"], "\n"),
        "arguments": strings.Join(arguments, ",\n    "),
        "inits": strings.Join(templateVals["inits"], "\n    "),
        "finish": strings.Join(templateVals["finish"], "\n"),
    }

    // add the dynamic placeholders: loads#, ops#, reduction#
    for _, key := range placeholders {
        templateParams[key] = strings.Join(templateVals[key], "\n        ")
    }

    // populate the template
    code := executeTemplate(templateSource, templateParams)

    return code, name
}

func initRand(templateVals map[string][]string) bool {
    templateVals["common"] = append(templateVals["common"], commonUrandGen)
    templateVals["inits"] = append(templateVals["inits"], initRandFunc)
    templateVals["finish"] = append(templateVals["finish"], finishRandFunc)
    return true
}

func executeTemplate(source string, params map[string]interface{}) string {
    templ := template.Must(template.New("float_ew").Parse(source))
    builder := new(strings.Builder)
    err := templ.Execute(builder, params)
    if err != nil {
        base.ValueError("%s", err.Error())
    }
    return builder.String()
}

func getKernelName() []string {
    // SKIPPED: Support for nvprof (NVPROF_ID defined in env)
    return []string{"kernel"}
}

