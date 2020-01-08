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

package data

import (
    "fmt"
    "fragata/arhat/base"
    "strings"
    "text/template"
)

//
//    Mnist
//

type Mnist struct {
    DataSetBase
    path string
    normalize bool
    symbol string
}

var mnistInitArgMap = base.ArgMap{
    "name": base.NewAnyArgOpt(""), // passthru
    "path": base.NewStringArgOpt("."),
    "normalize": base.NewBoolArgOpt(true),
}

func NewMnist(args ...interface{}) *Mnist {
    d := new(Mnist)
    d.Init(d, base.MakeArgs(args))
    return d
}

func(d *Mnist) Init(self DataSet, args base.Args) {
    d.DataSetBase.Init(self, args.Filter([]string{"name"}))
    args = cifar10InitArgMap.Expand(args)
    d.path = args["path"].(string)
    d.normalize = args["normalize"].(bool)

    index := d.Index()
    d.symbol = fmt.Sprintf("data_%d", index)
    imgSize := 28
    train := 
        NewDataIterator(
            fmt.Sprintf("train_%d", index),
            base.IntNone,
            10,
            []int{1, imgSize, imgSize},
            base.Float32, 
            "trainSet")
    d.SetIter("train", train)
    valid := 
        NewDataIterator(
            fmt.Sprintf("valid_%d", index),
            base.IntNone,
            10,
            []int{1, imgSize, imgSize},
            base.Float32, 
            "validSet")
    d.SetIter("valid", valid)
}

func(d *Mnist) ClassName() string {
    return "fragata.arhat.generators.data.Mnist"
}

func(d *Mnist) Declare() {
    b := d.Be()
    b.WriteLine("Mnist %s;", d.symbol)
    b.WriteLine("ArrayIterator %s;", d.TrainIter().Symbol())
    b.WriteLine("ArrayIterator %s;", d.ValidIter().Symbol())
}

var mnistInitCode = `
    {{.Data}}.Load("{{.Path}}", {{.Normalize}});
    int imgSize_{{.Index}} = {{.Data}}.ImgSize();
    {{.Train}}.Init(
        {{.Data}}.XTrain(),
        {{.Data}}.NumTrain(),
        imgSize_{{.Index}} * imgSize_{{.Index}},
        {{.Data}}.YTrain(),
        1,
        10,
        true);
    {{.Train}}.SetBsz({{.Bsz}});
    {{.Valid}}.Init(
        {{.Data}}.XTest(),
        {{.Data}}.NumTest(),
        imgSize_{{.Index}} * imgSize_{{.Index}},
        {{.Data}}.YTest(),
        1,
        10,
        true);
    {{.Valid}}.SetBsz({{.Bsz}});
`

var mnistInitTmpl = template.Must(template.New("mnistInit").Parse(mnistInitCode))

func(d *Mnist) Initialize() {
    btoa := func(b bool) string {
        if b {
            return "true"
        } else {
            return "false"
        }
    }
    data := map[string]interface{}{
        "Data": d.symbol,
        "Path": d.path,
        "Normalize": btoa(d.normalize),
        "Train": d.TrainIter().Symbol(),
        "Valid": d.ValidIter().Symbol(),
        "Bsz": d.Be().Bsz(),
        "Index": d.Index(),
    }
    var chunk strings.Builder
    err := mnistInitTmpl.Execute(&chunk, data)
    if err != nil {
        base.RuntimeError("%s", err.Error())
    }
    d.be.WriteChunk(chunk.String())
}

//
//    Cifar10
//

type Cifar10 struct {
    DataSetBase
    path string
    normalize bool
    contrastNormalize bool
    whiten bool
    padClasses bool
    symbol string
}

var cifar10InitArgMap = base.ArgMap{
    "name": base.NewAnyArgOpt(""), // passthru
    "path": base.NewStringArgOpt("."),
    "normalize": base.NewBoolArgOpt(true),
    "contrast_normalize": base.NewBoolArgOpt(false),
    "whiten": base.NewBoolArgOpt(false),
    "pad_classes": base.NewBoolArgOpt(false),
}

func NewCifar10(args ...interface{}) *Cifar10 {
    d := new(Cifar10)
    d.Init(d, base.MakeArgs(args))
    return d
}

func(d *Cifar10) Init(self DataSet, args base.Args) {
    d.DataSetBase.Init(self, args.Filter([]string{"name"}))
    args = cifar10InitArgMap.Expand(args)
    d.path = args["path"].(string)
    d.normalize = args["normalize"].(bool)
    d.contrastNormalize = args["contrast_normalize"].(bool)
    d.whiten = args["whiten"].(bool)
    d.padClasses = args["pad_classes"].(bool)

    index := d.Index()
    d.symbol = fmt.Sprintf("data_%d", index)
    var nclass int
    if d.padClasses {
        nclass = 16
    } else {
        nclass = 10
    }
    imgSize := 32
    train := 
        NewDataIterator(
            fmt.Sprintf("train_%d", index),
            base.IntNone,
            nclass,
            []int{3, imgSize, imgSize},
            base.Float32, 
            "trainSet")
    d.SetIter("train", train)
    valid := 
        NewDataIterator(
            fmt.Sprintf("valid_%d", index),
            base.IntNone,
            nclass,
            []int{3, imgSize, imgSize},
            base.Float32, 
            "validSet")
    d.SetIter("valid", valid)
}

func(d *Cifar10) ClassName() string {
    return "fragata.arhat.generators.data.Cifar10"
}

func(d *Cifar10) Declare() {
    b := d.Be()
    b.WriteLine("Cifar10 %s;", d.symbol)
    b.WriteLine("ArrayIterator %s;", d.TrainIter().Symbol())
    b.WriteLine("ArrayIterator %s;", d.ValidIter().Symbol())
}

var cifar10InitCode = `
    {{.Data}}.Load("{{.Path}}", {{.Normalize}}, {{.ContrastNormalize}}, {{.Whiten}}, {{.PadClasses}});
    int imgSize_{{.Index}} = {{.Data}}.ImgSize();
    {{.Train}}.Init(
        {{.Data}}.XTrain(),
        {{.Data}}.NumTrain(),
        3 * imgSize_{{.Index}} * imgSize_{{.Index}},
        {{.Data}}.YTrain(),
        1,
        {{.Nclass}},
        true);
    {{.Train}}.SetBsz({{.Bsz}});
    {{.Valid}}.Init(
        {{.Data}}.XTest(),
        {{.Data}}.NumTest(),
        3 * imgSize_{{.Index}} * imgSize_{{.Index}},
        {{.Data}}.YTest(),
        1,
        {{.Nclass}},
        true);
    {{.Valid}}.SetBsz({{.Bsz}});
`

var cifar10InitTmpl = template.Must(template.New("cifar10Init").Parse(cifar10InitCode))

func(d *Cifar10) Initialize() {
    btoa := func(b bool) string {
        if b {
            return "true"
        } else {
            return "false"
        }
    }
    var nclass int
    if d.padClasses {
        nclass = 16
    } else {
        nclass = 10
    }
    data := map[string]interface{}{
        "Data": d.symbol,
        "Path": d.path,
        "Normalize": btoa(d.normalize),
        "ContrastNormalize": btoa(d.contrastNormalize),
        "Whiten": btoa(d.whiten),
        "PadClasses": btoa(d.padClasses),
        "Nclass": nclass,
        "Train": d.TrainIter().Symbol(),
        "Valid": d.ValidIter().Symbol(),
        "Bsz": d.Be().Bsz(),
        "Index": d.Index(),
    }
    var chunk strings.Builder
    err := cifar10InitTmpl.Execute(&chunk, data)
    if err != nil {
        base.RuntimeError("%s", err.Error())
    }
    d.be.WriteChunk(chunk.String())
}

