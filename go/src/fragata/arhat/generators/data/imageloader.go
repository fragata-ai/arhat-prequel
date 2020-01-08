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
    "fragata/arhat/generators"
    "strings"
    "text/template"
)

//
//    CannedImageData
//

type CannedImageData struct {
    DataSetBase
    path string
    shape []int
    nclass int
    pixelMean []int
    sourceRange []float64
    targetRange []float64
}

var cannedImageDataInitArgMap = base.ArgMap{
    "name": base.NewAnyArgOpt(""), // passthru
    "path": base.NewStringArgOpt("."),
    "shape": base.NewIntListArg(),
    "nclass": base.NewIntArg(),
    "pixel_mean": base.NewIntListArgOpt(nil),
    "source_range": base.NewFloatListArgOpt(nil),
    "target_range": base.NewFloatListArgOpt(nil),
}

func NewCannedImageData(args ...interface{}) *CannedImageData {
    d := new(CannedImageData)
    d.Init(d, base.MakeArgs(args))
    return d
}

func(d *CannedImageData) Init(self DataSet, args base.Args) {
    d.DataSetBase.Init(self, args.Filter([]string{"name"}))
    args = cannedImageDataInitArgMap.Expand(args)
    d.path = args["path"].(string)
    d.shape = base.ToIntList(args["shape"])
    d.nclass = args["nclass"].(int)
    d.pixelMean = base.ToIntList(args["pixel_mean"])
    if len(d.pixelMean) != 0 {
        if len(d.pixelMean) != 3 {
            base.InvalidArgument("pixel_mean")
        }
        for _, mean := range d.pixelMean {
            if mean < 0 || mean > 255 {
                base.InvalidArgument("pixel_mean")
            }
        }
    }
    d.sourceRange = base.ToFloatList(args["source_range"])
    d.targetRange = base.ToFloatList(args["target_range"])
    if len(d.sourceRange) != 0 {
        if len(d.sourceRange) != 2 {
            base.InvalidArgument("source_range")
        }
        if len(d.targetRange) == 0 {
            base.MissingArgument("target_range")
        }
    }
    if len(d.targetRange) != 0 {
        if len(d.targetRange) != 2 {
            base.InvalidArgument("target_range")
        }
        if len(d.sourceRange) == 0 {
            base.MissingArgument("source_range")
        }
    }
    index := d.Index()
    train := 
        NewDataIterator(
            fmt.Sprintf("train_%d", index),
            base.IntNone,
            d.nclass,
            d.shape,
            base.Float32, 
            "trainSet")
    d.SetIter("train", train)
    valid := 
        NewDataIterator(
            fmt.Sprintf("valid_%d", index),
            base.IntNone,
            d.nclass,
            d.shape,
            base.Float32, 
            "validSet")
    d.SetIter("valid", valid)
}

func(d *CannedImageData) ClassName() string {
    return "fragata.arhat.generators.data.CannedImageData"
}

func(d *CannedImageData) Declare() {
    b := d.Be()
    b.WriteLine("CannedImageLoader %s;", d.TrainIter().Symbol())
    b.WriteLine("CannedImageLoader %s;", d.ValidIter().Symbol())
}

var cannedImageDataInitCode = `
    {{.Train}}.Init("{{.TrainX}}", "{{.TrainY}}", true);
    {{- if .PixelMean}}
    {{.Train}}.RgbMeanSubtract({{.PixelMean0}}, {{.PixelMean1}}, {{.PixelMean2}});
    {{- end}}
    {{- if .Normalize}}
    {{.Train}}.ValueNormalize({{.SourceRange0}}, {{.SourceRange1}}, {{.TargetRange0}}, {{.TargetRange1}});
    {{- end}}
    {{.Train}}.SetBsz({{.Bsz}});
    {{.Valid}}.Init("{{.ValidX}}", "{{.ValidY}}", true);
    {{- if .PixelMean}}
    {{.Valid}}.RgbMeanSubtract({{.PixelMean0}}, {{.PixelMean1}}, {{.PixelMean2}});
    {{- end}}
    {{- if .Normalize}}
    {{.Valid}}.ValueNormalize({{.SourceRange0}}, {{.SourceRange1}}, {{.TargetRange0}}, {{.TargetRange1}});
    {{- end}}
    {{.Valid}}.SetBsz({{.Bsz}});
`

var cannedImageDataInitTmpl = 
    template.Must(template.New("cannedImageDataInit").Parse(cannedImageDataInitCode))

func(d *CannedImageData) Initialize() {
    data := map[string]interface{}{
        "Train": d.TrainIter().Symbol(),
        "TrainX": d.path + "/train_x.dat",
        "TrainY": d.path + "/train_y.dat",
        "Valid": d.ValidIter().Symbol(),
        "ValidX": d.path + "/valid_x.dat",
        "ValidY": d.path + "/valid_y.dat",
        "Bsz": d.Be().Bsz(),
    }
    if d.pixelMean != nil {
        data["PixelMean"] = true
        data["PixelMean0"] = d.pixelMean[0]
        data["PixelMean1"] = d.pixelMean[1]
        data["PixelMean2"] = d.pixelMean[2]
    }
    if d.sourceRange != nil {
        formatFloat32 := generators.FormatFloat32
        data["Normalize"] = true
        data["SourceRange0"] = formatFloat32(d.sourceRange[0])
        data["SourceRange1"] = formatFloat32(d.sourceRange[1])
        data["TargetRange0"] = formatFloat32(d.targetRange[0])
        data["TargetRange1"] = formatFloat32(d.targetRange[1])
    }
    var chunk strings.Builder
    err := cannedImageDataInitTmpl.Execute(&chunk, data)
    if err != nil {
        base.RuntimeError("%s", err.Error())
    }
    d.be.WriteChunk(chunk.String())
}

