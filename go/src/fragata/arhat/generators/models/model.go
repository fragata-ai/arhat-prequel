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

package models

import (
    "fmt"
    "fragata/arhat/backends"
    "fragata/arhat/base"
    "fragata/arhat/generators"
    "fragata/arhat/generators/callbacks"
    "fragata/arhat/generators/data"
    "fragata/arhat/layers"
    "fragata/arhat/models"
    "fragata/arhat/optimizers"
    "fragata/arhat/transforms"
    "strings"
)

//
//    Model
//

type Model struct {
    models.Model
    be generators.Generator
}

func NewModel(argLayers interface{}, args ...interface{}) *Model {
    m := new(Model)
    m.Init(m, argLayers, base.MakeArgs(args))
    return m
}

func(m *Model) Init(self base.Object, argLayers interface{}, args base.Args) {
    m.Model.Init(self, argLayers, args)
    m.be = backends.Be().(generators.Generator)
}

func(m *Model) ClassName() string {
    return "arhat.generators.models.Model"
}

func(m *Model) Initialize(inObj []int, cost layers.Cost) {
    if m.Initialized() {
        return
    }
    b := m.be
    b.EnterInit()
    id := m.makeInitializeId()
    b.WriteLine("void %s() {", id)
    b.Indent(1)
    m.Model.Initialize(inObj, cost)
    b.Indent(-1)
    b.WriteLine("}")
    b.WriteLine("")
    b.ExitInit()
    b.WriteLine("%s();", id)
}

func(m *Model) Fprop(x []backends.Tensor, inference bool) []backends.Tensor {
    b := m.be
    b.PushCode()
    id := m.makeFpropId()
    b.WriteLine("// model=%s inference=%t", m.Name(), inference)
    b.WriteLine("void %s() {", id)
    b.Indent(1)
    res := m.Layers().Fprop(x, inference, 0.0)
    b.Indent(-1)
    b.WriteLine("}")
    b.WriteLine("")
    b.PopCode()
    b.WriteLine("%s();", id)
    return res
}

func(m *Model) Bprop(delta []backends.Tensor) []backends.Tensor {
    b := m.be
    b.PushCode()
    id := m.makeBpropId()
    b.WriteLine("// model=%s", m.Name())
    b.WriteLine("void %s() {", id)
    b.Indent(1)
    res := m.Layers().Bprop(delta, 1.0, 0.0)
    b.Indent(-1)
    b.WriteLine("}")
    b.WriteLine("")
    b.PopCode()
    b.WriteLine("%s();", id)
    return res
}

func(m *Model) Fit(
        dataset *data.DataIterator, 
        cost layers.Cost, 
        optimizer optimizers.Optimizer, 
        numEpochs int, 
        callbks *callbacks.Callbacks) {
    b := m.be
    b.PushCode()
    id := m.makeFitId()
    b.WriteLine("// model=%s", m.Name())
    b.WriteLine("void %s(DataIterator *dataset, Callbacks *callbacks) {", id)
    b.Indent(1)

    scheduleRef := "nullptr"
    if schedule := optimizer.Schedule(); schedule != nil {
        ctor := schedule.Construct("schedule")
        for _, line := range ctor {
            b.WriteLine("%s", line)
        }
        scheduleRef = "&schedule"
    }

    m.Initialize(dataset.Shape(), cost)

    b.WriteLine("callbacks->OnTrainBegin();")
    b.WriteLine("int numEpochs = %d;", numEpochs)
    b.WriteLine("int epoch = 0;")
    b.WriteLine("while (epoch < numEpochs && !callbacks->Finished()) {")
    b.Indent(1)
    b.WriteLine("callbacks->OnEpochBegin(epoch);")
    m.epochFit(dataset, optimizer, scheduleRef, "epoch")
    b.WriteLine("callbacks->OnEpochEnd(epoch);")
    b.WriteLine("epoch++;")
    b.Indent(-1)
    b.WriteLine("}")
    b.WriteLine("callbacks->OnTrainEnd();")

    b.Indent(-1)
    b.WriteLine("}")
    b.WriteLine("")
    b.PopCode()

    b.WriteLine("%s(&%s, &%s);", id, dataset.Symbol(), callbks.Symbol())
}

func(m *Model) epochFit(
        dataset *data.DataIterator, 
        optimizer optimizers.Optimizer,
        scheduleRef string,
        epochSymbol string) {
    b := m.be
    b.PushCode()
    id := m.makeEpochFitId()
    b.WriteLine("// model=%s", m.Name())
    sig := fmt.Sprintf(
        "void %s(DataIterator *dataset, Callbacks *callbacks, Schedule *schedule, int epoch) {",
            id)
    m.writeLongSig(sig)
    b.Indent(1)

    b.WriteLine("float totalCost = 0.0;")

    layersToOptimize := m.LayersToOptimize()
    optimizer.Reset(layersToOptimize)

    // iterate through minibatches of the dataset
    b.WriteLine("int mbIdx = 0;")
    m.reset(dataset)
    m.start(dataset)
    b.WriteLine("while (%s) {", m.iter(dataset))
    b.Indent(1)

    b.WriteLine("callbacks->OnMinibatchBegin(epoch, mbIdx);")

    cost := m.Cost()
    x := []backends.Tensor{dataset.X()}
    t := []backends.Tensor{dataset.Y()}

    x = m.Fprop(x, false)

    b.WriteLine("// GetCost: cost=%s", cost.Name())
    c := cost.GetCost(x, t)
    b.WriteLine("totalCost += %s;", generators.FormatScalar(c))

    // deltas back propagate through layers
    // for every layer in reverse except the 0th one
    b.WriteLine("// GetErrors: cost=%s", cost.Name())
    delta := cost.GetErrors(x, t)

    m.Bprop(delta)
    m.optimize(optimizer, layersToOptimize, "epoch")

    b.WriteLine("callbacks->OnMinibatchEnd(epoch, mbIdx);")
    b.WriteLine("mbIdx++;")

    b.Indent(-1)
    b.WriteLine("}") // while

    // now we divide total cost by the number of batches,
    // so it was never total cost, but sum of averages
    // across all the minibatches we trained on
    b.WriteLine("totalCost /= float(dataset->Nbatches());")

    // TODO: Revise this (replace with passing totalCost to callback)
    b.WriteLine("printf(\"Epoch %%d: total cost %%g\\n\", epoch, totalCost);")

    b.Indent(-1)
    b.WriteLine("}")
    b.WriteLine("")
    b.PopCode()

    b.WriteLine("%s(dataset, callbacks, %s, %s);", id, scheduleRef, epochSymbol)
}

func(m *Model) Eval(dataset *data.DataIterator, metric transforms.Metric) []backends.Value {
    b := m.be
    b.PushCode()
    id := m.makeEvalId()
    b.WriteLine("// model=%s", m.Name())
    b.WriteLine("void %s(DataIterator *dataset, float *runningError) {", id)
    b.Indent(1)

    m.Initialize(dataset.Shape(), nil)
    numMetrics := len(metric.MetricNames())
    for i := 0; i < numMetrics; i++ {
        b.WriteLine("runningError[%d] = 0.0f;", i)
    }
    m.reset(dataset)
    b.WriteLine("int nprocessed = 0;")
    b.WriteLine("int ndata = dataset->Ndata();")
    // SKIPPED: Support for dataset.SeqLength() - semantics and usage not yet clarified

    b.WriteLine("dataset->Start();")
    b.WriteLine("while (%s) {", m.iter(dataset))
    b.Indent(1)

    x := []backends.Tensor{dataset.X()}
    t := []backends.Tensor{dataset.Y()}

    x = m.Fprop(x, true)

    bsz := b.Bsz()
    nsteps := x[0].Shape()[1] / bsz

    b.WriteLine("int maxBsz = ndata - nprocessed;")
    b.WriteLine("int bsz = (%d < maxBsz) ? %d : maxBsz;", bsz, bsz)

    b.WriteLine("int size = %d * bsz;", nsteps)
    e := metric.Call(x, t)
    for i, ei := range e {
        b.WriteLine("runningError[%d] += %s;", i, b.GetMetricSum(ei, "0", "size"))
    }
    b.WriteLine("nprocessed += size;")

    b.Indent(-1)
    b.WriteLine("}") // while

    for i := 0; i < numMetrics; i++ {
        b.WriteLine("runningError[%d] /= float(nprocessed);", i)
    }

    b.Indent(-1)
    b.WriteLine("}")
    b.WriteLine("")
    b.PopCode()

    errors := fmt.Sprintf("errors_%d", b.MakeIndex("var"))
    b.WriteLine("float %s[%d];", errors, numMetrics);
    b.WriteLine("%s(&%s, %s);", id, dataset.Symbol(), errors)

    result := make([]backends.Value, numMetrics)
    for i := 0; i < numMetrics; i++ {
        result[i] = generators.NewFloatSymbol(fmt.Sprintf("%s[%d]", errors, i))
    }
    return result
}

func(m *Model) GetOutputs(dataset *data.DataIterator, writer data.DataWriter) {
    b := m.be
    b.PushCode()
    id := m.makeGetOutputId()

    b.WriteLine("// model=%s", m.Name())
    b.WriteLine("void %s(DataIterator *dataset, DataWriter *writer) {", id)
    b.Indent(1)

    m.Initialize(dataset.Shape(), nil)
    m.reset(dataset)
    b.WriteLine("int ndata = dataset->Ndata();")
    b.WriteLine("int idx = 0;")

    b.WriteLine("dataset->Start();")
    b.WriteLine("while (%s) {", m.iter(dataset))
    b.Indent(1)

    x := []backends.Tensor{dataset.X()}

    x = m.Fprop(x, true)

    base.AssertMsg(len(x) == 1, "Can not Getutputs with Branch terminal")

    shape := x[0].Shape()
    dim1 := shape[1]

    b.WriteLine("int start = idx * %d;", dim1)
    b.WriteLine("int stop = (idx + 1) * %d;", dim1)
    b.WriteLine("stop = (stop > ndata) ? ndata : stop;")

    // writer knows tensor shape, hence only actual batch size is passed
    // WriteBatch must copy data to host and transpose taking batch size in account
    b.WriteLine("writer->WriteBatch(%s, stop - start);", b.FormatBufferRef(x[0], false)) 

    b.WriteLine("idx++;")

    b.Indent(-1)
    b.WriteLine("}") // while

    // SKIPPED: Handle the recurrent case (dim1 > bsz)

    b.Indent(-1)
    b.WriteLine("}")
    b.WriteLine("")
    b.PopCode()

    b.WriteLine("%s(&%s, &%s);", id, dataset.Symbol(), writer.Symbol())
}

func(m *Model) LoadParams(paramPath string) {
    base.AssertMsg(m.Initialized(), "Model must be initialized prior to LoadParams")

    b := m.be
    b.PushCode()
    id := m.makeLoadParamsId()

    b.WriteLine("// model=%s", m.Name())
    b.WriteLine("void %s(const char *paramPath) {", id)
    b.Indent(1)

    b.WriteLine("MemoryReader reader;")
    b.WriteLine(`reader.Load("%s");`, paramPath)
    r := NewParamReader(b, "reader")
    m.ReadParams(r)

    b.Indent(-1)
    b.WriteLine("}")
    b.WriteLine("")
    b.PopCode()

    b.WriteLine(`%s("%s");`, id, paramPath)
}

func(m *Model) SaveParams(paramPath string) {
    b := m.be
    b.PushCode()
    id := m.makeSaveParamsId()

    b.WriteLine("// model=%s", m.Name())
    b.WriteLine("void %s(const char *paramPath) {", id)
    b.Indent(1)

    b.WriteLine("MemoryWriter writer;")
    w := NewParamWriter(b, "writer")
    m.WriteParams(w)
    b.WriteLine(`writer.Save("%s");`, paramPath)

    b.Indent(-1)
    b.WriteLine("}")
    b.WriteLine("")
    b.PopCode()

    b.WriteLine(`%s("%s");`, id, paramPath)
}

func(m *Model) optimize(
        optimizer optimizers.Optimizer, 
        layerList []layers.Layer, 
        epochSymbol string) {
    b := m.be
    b.PushCode()
    id := m.makeOptimizeId(optimizer)
    b.WriteLine("// optimizer=%s", optimizer.Name())
    b.WriteLine("void %s(Schedule *schedule, int epoch) {", id)
    b.Indent(1)
    optimizer.Optimize(layerList, generators.NewIntSymbol("epoch"))
    b.Indent(-1)
    b.WriteLine("}")
    b.WriteLine("")
    b.PopCode()
    b.WriteLine("%s(schedule, %s);", id, epochSymbol)
}

func(m *Model) reset(dataset *data.DataIterator) {
    dataset.X().Fill(0.0)
    dataset.Y().Fill(0.0)
    m.be.WriteLine("dataset->Reset();")
}

func(m *Model) start(dataset *data.DataIterator) {
    m.be.WriteLine("dataset->Start();")
}

func(m *Model) iter(dataset *data.DataIterator) string {
    be := m.be
    xref := be.FormatBufferRef(dataset.X(), true)
    yref := be.FormatBufferRef(dataset.Y(), true)
    return fmt.Sprintf("dataset->Iter(%s, %s)", xref, yref)
}

func(m *Model) writeLongSig(sig string) {
    b := m.be
    parts := strings.SplitN(sig, "(", 2)
    part0 := parts[0]
    parts = strings.Split(parts[1], ",")
    if len(parts) <= 3 {
        b.WriteLine("%s", sig)
    } else {
        b.WriteLine(part0+"(")
        b.Indent(2)
        last := len(parts) - 1
        for _, line := range parts[:last] {
            b.WriteLine("%s,", strings.TrimSpace(line))
        }
        b.WriteLine("%s", strings.TrimSpace(parts[last]))
        b.Indent(-2)
    }
}

func(m *Model) makeInitializeId() string {
    index := m.be.MakeIndex("model.initialize")
    return fmt.Sprintf("InitializeModel_%d", index)
}

func(m *Model) makeFpropId() string {
    index := m.be.MakeIndex("model.fprop")
    return fmt.Sprintf("FpropModel_%d", index)
}

func(m *Model) makeBpropId() string {
    index := m.be.MakeIndex("model.bprop")
    return fmt.Sprintf("BpropModel_%d", index)
}

func(m *Model) makeFitId() string {
    index := m.be.MakeIndex("model.fit")
    return fmt.Sprintf("FitModel_%d", index)
}

func(m *Model) makeEpochFitId() string {
    index := m.be.MakeIndex("model.epoch_fit")
    return fmt.Sprintf("EpochFitModel_%d", index)
}

func(m *Model) makeEvalId() string {
    index := m.be.MakeIndex("model.eval")
    return fmt.Sprintf("EvalModel_%d", index)
}

func(m *Model) makeGetOutputId() string {
    index := m.be.MakeIndex("model.get_output")
    return fmt.Sprintf("GetOutputModel_%d", index)
}

func(m *Model) makeLoadParamsId() string {
    index := m.be.MakeIndex("model.load_params")
    return fmt.Sprintf("LoadParams_%d", index)
}

func(m *Model) makeSaveParamsId() string {
    index := m.be.MakeIndex("model.save_params")
    return fmt.Sprintf("SaveParams_%d", index)
}

func(m *Model) makeOptimizeId(optimizer optimizers.Optimizer) string {
    index := m.be.MakeIndex("model.optimize")
    return fmt.Sprintf("Optimize_%d", index)
}

//
//    ParamReader
//

type ParamReader struct {
    be generators.Generator
    symbol string
}

func NewParamReader(be generators.Generator, symbol string) *ParamReader {
    return &ParamReader{be: be, symbol: symbol}
}

func(r *ParamReader) Read(x backends.Tensor) {
    b := r.be
    size := x.Size() * x.Dtype().ItemSize()
    b.WriteLine("%s.Read(%d, %s);", r.symbol, size, b.FormatBufferRef(x, false))
}

//
//    ParamWriter
//

type ParamWriter struct {
    be generators.Generator
    symbol string
}

func NewParamWriter(be generators.Generator, symbol string) *ParamWriter {
    return &ParamWriter{be: be, symbol: symbol}
}

func(r *ParamWriter) Write(x backends.Tensor) {
    b := r.be
    size := x.Size() * x.Dtype().ItemSize()
    b.WriteLine("%s.Write(%d, %s);", r.symbol, size, b.FormatBufferRef(x, false))
}

