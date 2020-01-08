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

package cpu

import (
    "fmt"
    "fragata/arhat/backends"
    "fragata/arhat/base"
    "fragata/arhat/generators"
    "fragata/arhat/generators/acc"
    "os"
    "path"
    "strings"
)

//
//    Global buffer memory
//

type CpuDeviceAllocation struct {
    index int
    nbytes int
    offset int
}

func NewCpuDeviceAllocation(index int, nbytes int, offset int) *CpuDeviceAllocation {
    return &CpuDeviceAllocation{index, nbytes, offset}
}

func(a *CpuDeviceAllocation) Add(offset int) acc.DeviceAllocation {
    nbytes := a.nbytes - offset
    base.Assert(nbytes >= 0)
    return NewCpuDeviceAllocation(a.index, nbytes, a.offset+offset)
}

//
//    Kernel
//

type Kernel struct {
    lib *CpuGenerator
    name string
    code string
}

func NewKernel(lib *CpuGenerator, name string, code string) *Kernel {
    return &Kernel{lib, name, code}
}

//
//    CpuGenerator
//

type CpuGenerator struct {
    acc.AccGeneratorBase
    scratchData acc.DeviceAllocation
    scratchSize int
    scratchOffset int
    bench bool
    buffers []*CpuDeviceAllocation
    nextBufferIndex int
    nextKernelIndex int
    buildMainCpp bool
    filePrefix string
    hostNamespace string
    kernelPrefix string
    kernelCache map[string]*Kernel
}

func NewCpuGenerator(
        rngSeed int,
        defaultDtype base.Dtype,
        stochasticRound int,
        bench bool,
        scratchSize int,
        histBins int,
        histOffset int,
        compatMode backends.CompatMode) *CpuGenerator {
    b := new(CpuGenerator)
    b.Init(
        b,
        rngSeed,
        defaultDtype,
        stochasticRound,
        bench,
        scratchSize,
        histBins,
        histOffset,
        compatMode)
    return b
}

func(b *CpuGenerator) Init(
        self generators.Generator,
        rngSeed int,
        defaultDtype base.Dtype,
        stochasticRound int,
        bench bool,
        scratchSize int,
        histBins int,
        histOffset int,
        compatMode backends.CompatMode) {
    defaultDtype = base.ResolveDtype(defaultDtype, base.Float32)
    stochasticRound = base.ResolveInt(stochasticRound, 0)
    scratchSize = base.ResolveInt(scratchSize, 0)
    histBins = base.ResolveInt(histBins, 64)
    histOffset = base.ResolveInt(histOffset, -48)

    if defaultDtype != base.Float32 {
        base.ValueError("Default data type for cpu backend must be float32")
    }

    // AccGeneratorBase.Init allocates device memory buffers;
    //     therefore buffer index must be initialized before
    b.nextBufferIndex = 0

    b.AccGeneratorBase.Init(
        self,
        rngSeed,
        defaultDtype,
        stochasticRound,
        histBins,
        histOffset,
        compatMode)

    // attributes
    b.scratchData = nil
    b.scratchSize = scratchSize
    b.scratchOffset = 0
    b.bench = bench

    b.nextKernelIndex = 1

    b.buildMainCpp = true

    b.kernelCache = make(map[string]*Kernel)
}

func(b *CpuGenerator) ConfigureCodeOutput(
        buildMainCpp bool,
        filePrefix string, 
        hostNamespace string, 
        kernelPrefix string) {
    b.buildMainCpp = buildMainCpp
    b.filePrefix = filePrefix
    b.hostNamespace = hostNamespace
    b.kernelPrefix = kernelPrefix
}

func(b *CpuGenerator) ScratchBufferInit() {
    b.scratchOffset = 0
}

func(b *CpuGenerator) ScratchBufferOffset(size int) acc.DeviceAllocation {
    if size & 127 != 0 {
        size += 128 - (size & 127)
    }

    if size + b.scratchOffset > b.scratchSize {
        base.RuntimeError(
            "CpuGenerator.scratchSize(%d) is too small for this operation(%d, %d)",
                b.scratchSize, size, b.scratchOffset)
    }

    if b.scratchData == nil {
        b.scratchData = b.MemAlloc(b.scratchSize)
    }

    data := b.scratchData.Add(b.scratchOffset)
    b.scratchOffset += size

    return data
}

func(b *CpuGenerator) SetScratchSize(args ...int) {
    totalSize := 0
    for _, size := range args {
        if size & 127 != 0 {
            size += 128 - (size & 127)
        }
        totalSize += size
    }

    if totalSize > b.scratchSize {
        b.scratchSize = totalSize
    }
}

func(b *CpuGenerator) LookupKernel(key string) *Kernel {
    kernel, ok := b.kernelCache[key]
    if !ok {
        return nil
    }
    // debugging hooks can be placed here
    return kernel
}

func(b *CpuGenerator) RegisterKernel(key string, kernel *Kernel) {
    b.kernelCache[key] = kernel
}

// Backends methods

func(b *CpuGenerator) RngNormal(out backends.Tensor, loc float64, scale float64, size []int) {
    b.WriteLine("RngNormal((float *)%s, %s, %s, %d);",
        formatBufferRef(out.(*acc.AccTensor).AccData().(*CpuDeviceAllocation), true),
        formatFloat32(loc),
        formatFloat32(scale),
        base.IntsProd(size))
}

func(b *CpuGenerator) RngUniform(out backends.Tensor, low float64, high float64, size []int) {
    b.WriteLine("RngUniform((float *)%s, %s, %s, %d);",
        formatBufferRef(out.(*acc.AccTensor).AccData().(*CpuDeviceAllocation), true),
        formatFloat32(low),
        formatFloat32(high),
        base.IntsProd(size))
}

func(b *CpuGenerator) CompoundDot(
        x backends.Tensor,
        y backends.Tensor,
        z backends.Tensor,
        alpha float64,
        beta float64,
        relu bool,
        bsum backends.Tensor) backends.Tensor {
    cpuBsum, _ := bsum.(*acc.AccTensor) // may be nil
    return b.CompoundDotCpu(
        x.(*acc.AccTensor),
        y.(*acc.AccTensor),
        z.(*acc.AccTensor),
        alpha,
        beta,
        relu,
        cpuBsum,
        1, 
        nil)
}

func(b *CpuGenerator) CompoundDotCpu(
        x *acc.AccTensor,
        y *acc.AccTensor,
        z *acc.AccTensor,
        alpha float64, 
        beta float64,
        relu bool,
        bsum *acc.AccTensor,
        repeat int,
        size []int) *acc.AccTensor {
    b.blasDot(x, y, z, alpha, beta)
    if relu {
        z.Assign(b.Maximum(z, b.Float(0.0)))
    }
    if bsum != nil {
        bsum.Assign(b.Sum(z, 1))
    }
    return z
}

func(b *CpuGenerator) blasDot(
        x *acc.AccTensor,
        y *acc.AccTensor,
        z *acc.AccTensor,
        alpha float64, 
        beta float64) {
    ldx := base.IntsMax(x.Strides())
    ldy := base.IntsMax(y.Strides())
    ldz := base.IntsMax(z.Strides())

    opX := 'n'
    if x.IsTrans() {
        opX = 't'
    }
    opY := 'n'
    if y.IsTrans() {
        opY = 't'
    }

    xshape := x.Shape()
    yshape := y.Shape()
    zshape := z.Shape()

    m := xshape[0]
    n := yshape[1]
    k := xshape[1]

    xDtype := x.Dtype()
    falpha := formatFloat32(alpha)
    fbeta := formatFloat32(beta)
    fmtData := func(a acc.DeviceAllocation) string {
        return formatBufferRef(a.(*CpuDeviceAllocation), true)
    }

    // Swap X and Y to map from C order to Fortran
    switch {
    case xDtype == base.Float32:
        xdata := fmtData(x.AccData())
        ydata := fmtData(y.AccData())
        zdata := fmtData(z.AccData())

        if n != 1 || (opX == 't' && opY == 'n') {
            b.writeLongStmt(
                "BlasSgemm('%c', '%c', %d, %d, %d,#"+
                    "%s, (float *)%s, %d, (float *)%s, %d,#"+
                    "%s, (float *)%s, %d);",
                        opY, opX, n, m, k, falpha, ydata, ldy, xdata, ldx, fbeta, zdata, ldz)
        } else {
            b.writeLongStmt(
                "BlasSgemv('t', %d, %d,#"+
                    "%s, (float *)%s, %d, (float *)%s, %d,#"+
                    "%s, (float *)%s, %d);",
                        k, m, falpha, xdata, k, ydata, ldy, fbeta, zdata, ldz)
        }

    case xDtype == base.Float16:
        xtemp := b.BufMalloc([]int{xshape[0], xshape[1]*2})
        ytemp := b.BufMalloc([]int{yshape[0], yshape[1]*2})
        ztemp := b.BufMalloc([]int{zshape[0], zshape[1]*2})

        xfp32 := 
            acc.NewAccTensor(
                b, 
                xshape, 
                base.Float32, 
                "",              // name
                true,            // persistValues
                nil,             // tbase
                xtemp.AccData(),
                x.Strides(),
                nil,             // takeArray
                x.IsTrans(),
                base.IntNone)    // rounding
        yfp32 := 
            acc.NewAccTensor(
                b, 
                yshape, 
                base.Float32, 
                "",              // name
                true,            // persistValues
                nil,             // tbase
                ytemp.AccData(),
                y.Strides(),
                nil,             // takeArray
                y.IsTrans(),
                base.IntNone)    // rounding
        zfp32 := 
            acc.NewAccTensor(
                b, 
                zshape, 
                base.Float32, 
                "",              // name
                true,            // persistValues
                nil,             // tbase
                ztemp.AccData(),
                z.Strides(),
                nil,             // takeArray
                z.IsTrans(),
                base.IntNone)    // rounding

        xfp32.Assign(x)
        yfp32.Assign(y)
        zfp32.Assign(z)

        xdata := fmtData(xfp32.AccData())
        ydata := fmtData(yfp32.AccData())
        zdata := fmtData(zfp32.AccData())

        if n != 1 || (opX == 't' && opY == 'n') {
            b.writeLongStmt(
                "BlasSgemm('%c', '%c', %d, %d, %d,#"+
                    "%s, (float *)%s, %d, (float *)%s, %d,#"+
                    "%s, (float *)%s, %d);",
                        opY, opX, n, m, k, falpha, ydata, ldy, xdata, ldx, fbeta, zdata, ldz)
        } else {
            b.writeLongStmt(
                "BlasSgemv('t', %d, %d,#"+
                    "%s, (float *)%s, %d, (float *)%s, %d,#"+
                    "%s, (float *)%s, %d);",
                        k, m, falpha, xdata, k, ydata, ldy, fbeta, zdata, ldz)
        }

        z.Assign(zfp32)
        b.BufFree()

    default:
        base.TypeError("Unsupported type for cublas gemm")
    }
}

func(b *CpuGenerator) MakeBinaryMask(out backends.Tensor, keepThresh float64) {
    // generate random number uniformly distributed between 0 and 1
    b.RngUniform(out, 0.0, 1.0, out.Shape())
    out.Assign(b.LessEqual(out, b.Float(keepThresh)))
}

func(b *CpuGenerator) Binarize(
        ary backends.Tensor, out backends.Tensor, stochastic bool) backends.Tensor {
    // TODO
    base.NotImplementedError()
    return nil
}

func(b *CpuGenerator) NewConvLayer(
        dtype base.Dtype, params *backends.ConvParams) backends.ConvLayer {
    return NewConvLayer(b, dtype, params)
}

func(b *CpuGenerator) NewDeconvLayer(
        dtype base.Dtype, params *backends.DeconvParams) backends.DeconvLayer {
    return NewDeconvLayer(b, dtype, params)
}

func(b *CpuGenerator) FpropConv(
        layer backends.ConvLayerBase,
        i backends.Tensor,
        f backends.Tensor,
        o backends.Tensor,
        x backends.Tensor,
        bias backends.Tensor,
        bsum backends.Tensor,
        alpha float64,
        beta float64,
        relu bool,
        brelu bool,
        slope float64) {
    base.Assert(layer.SizeI() == i.Size())
    base.Assert(layer.SizeF() == f.Size())
    base.Assert(layer.SizeO() == o.Size())

    cpuLayer := castConvLayerBase(layer)
    cpuLayer.FpropConv(i, f, o, x, bias, bsum, alpha, beta, relu, brelu, slope)
}

func(b *CpuGenerator) BpropConv(
        layer backends.ConvLayerBase,
        f backends.Tensor,
        e backends.Tensor,
        gradI backends.Tensor,
        x backends.Tensor,
        bias backends.Tensor,
        bsum backends.Tensor,
        alpha float64,
        beta float64,
        relu bool,
        brelu bool,
        slope float64) {
    base.Assert(layer.SizeF() == f.Size())
    base.Assert(layer.SizeO() == e.Size())
    base.Assert(layer.SizeI() == gradI.Size())

    cpuLayer := castConvLayerBase(layer)
    cpuLayer.BpropConv(e, f, gradI, x, bias, bsum, alpha, beta, relu, brelu, slope)
}

func(b *CpuGenerator) UpdateConv(
        layer backends.ConvLayerBase,
        i backends.Tensor,
        e backends.Tensor,
        gradF backends.Tensor,
        alpha float64,
        beta float64,
        gradBias backends.Tensor) {
    base.Assert(layer.SizeI() == i.Size())
    base.Assert(layer.SizeO() == e.Size())
    base.Assert(layer.SizeF() == gradF.Size())

    cpuLayer := castConvLayerBase(layer)
    cpuLayer.UpdateConv(i, e, gradF, alpha, beta, gradBias)
}

func castConvLayerBase(layer backends.ConvLayerBase) *ConvLayerBase {
    switch v := layer.(type) {
    case *ConvLayer:
        return &v.ConvLayerBase
    case *DeconvLayer:
        return &v.ConvLayerBase
    default:
        base.AssertionError("Invalid layer type")
        return nil
    }
}

func(b *CpuGenerator) NewLrnLayer(
        dtype base.Dtype, params *backends.LrnParams) backends.LrnLayer {
    return NewLrnLayer(b, dtype, params)
}

func(b *CpuGenerator) FpropLrn(
        layer backends.LrnLayer, 
        i backends.Tensor,
        o backends.Tensor,
        denom backends.Tensor,
        alpha float64,
        beta float64,
        ascale float64,
        bpower float64) {
    base.Assert(layer.SizeI() == i.Size())
    base.Assert(layer.SizeO() == o.Size())
    cpuLayer := layer.(*LrnLayer)
    cpuLayer.FpropLrn(i, o, denom, ascale, bpower)
}

func(b *CpuGenerator) BpropLrn(
        layer backends.LrnLayer, 
        i backends.Tensor,
        o backends.Tensor,
        e backends.Tensor,
        delta backends.Tensor,
        denom backends.Tensor,
        alpha float64,
        beta float64,
        ascale float64,
        bpower float64) {
    base.Assert(layer.SizeI() == i.Size())
    base.Assert(layer.SizeO() == e.Size())
    base.Assert(layer.SizeI() == delta.Size())
    cpuLayer := layer.(*LrnLayer)
    cpuLayer.BpropLrn(i, o, e, delta, denom, ascale, bpower)
}

func(b *CpuGenerator) NewPoolLayer(
        dtype base.Dtype, params *backends.PoolParams) backends.PoolLayer {
    return NewPoolLayer(b, dtype, params)
}

func(b *CpuGenerator) FpropPool(
        layer backends.PoolLayer, 
        i backends.Tensor, 
        o backends.Tensor, 
        argmax backends.Tensor, 
        alpha float64, 
        beta float64) {
    base.Assert(layer.SizeI() == i.Size())
    base.Assert(layer.SizeO() == o.Size())
    if layer.Op() == backends.PoolOpMax {
        base.AssertMsg(argmax != nil, "max pooling requires argmax buffer")
    }
    cpuLayer := layer.(*PoolLayer)
    cpuLayer.FpropPool(i, o, argmax, beta) // alpha is not used for CPU - why?
}

func(b *CpuGenerator) BpropPool(
        layer backends.PoolLayer, 
        i backends.Tensor, 
        o backends.Tensor, 
        argmax backends.Tensor, 
        alpha float64, 
        beta float64) {
    base.AssertMsg(layer.SizeI() == o.Size(),
        "mismatch between sizeI %d and O %d", layer.SizeI(), o.Size())
    base.AssertMsg(layer.SizeO() == i.Size(),
        "mismatch between sizeO %d and I %d", layer.SizeO(), i.Size())
    if layer.Op() == backends.PoolOpMax {
        base.AssertMsg(argmax != nil, "max pooling requires argmax buffer")
    }
    if argmax != nil {
        base.AssertMsg(layer.SizeO() == argmax.Size(), 
            "Pooling argmax size does not match input size!")
    }
    base.Assert(i.Dtype() == o.Dtype())
    cpuLayer := layer.(*PoolLayer)
    cpuLayer.BpropPool(i, o, argmax, alpha, beta)
}

func(b *CpuGenerator) NewReluLayer() backends.Layer {
    // not used with this backend
    return nil
}

func(b *CpuGenerator) FpropRelu(
        layer backends.Layer,
        x backends.Tensor, 
        slope float64) backends.Value {
   // maximum(x, 0) + slope * minimum(0, x)
    zero := b.Float(0.0)
    return b.Maximum(x, zero).Add(b.Float(slope).Mul(b.Minimum(x, zero)))
}

func(b *CpuGenerator) BpropRelu(
        layer backends.Layer,
        x backends.Tensor,
        errors backends.Tensor,
        deltas backends.Tensor,
        slope float64) backends.Value {
    // greater(x, 0) + slope * less(x, 0)
    zero := b.Float(0.0)
    return b.Greater(x, zero).Add(b.Float(slope).Mul(b.Less(x, zero)))
}

func(b *CpuGenerator) NewBatchNormLayer(inShape []int) backends.BatchNormLayer {
    // not used with this backend
    return nil
}

func(b *CpuGenerator) CompoundFpropBn(
        x backends.Tensor,
        xsum backends.Tensor,
        xvar backends.Tensor,
        gmean backends.Tensor,
        gvar backends.Tensor,
        gamma backends.Tensor,
        beta backends.Tensor,
        y backends.Tensor,
        eps float64,
        rho float64,
        computeBatchSum bool,
        accumbeta float64,
        relu bool,
        binary bool,
        inference bool,
        outputs backends.Tensor,
        later backends.BatchNormLayer) {
    if inference {
        // xhat = (x - gmean) / sqrt(gvar + eps)  # Op-tree only
        // y[] = y * accumbeta + xhat * gamma + beta
        xhat := x.Sub(gmean).Div(b.Sqrt(gvar.Add(b.Float(eps)))) // Op-tree only
        y.Assign(y.Mul(b.Float(accumbeta)).Add(xhat.Mul(gamma)).Add(beta))
        return
    }

    if computeBatchSum {
        xsum.Assign(b.Sum(x, 1))
    }

    xshape := x.Shape()

    // xvar[] = var(x, axis=1, binary=binary)
    // xsum[] = xsum / x.shape[1]
    xvar.Assign(b.Var(x, 1, binary))
    xsum.Assign(xsum.Div(b.Int(xshape[1]))) // reuse xsum instead of computing xmean

    // gmean[] = gmean * rho + (1.0 - rho) * xsum
    // gvar[] = gvar * rho + (1.0 - rho) * xvar
    one := b.Float(1.0)
    frho := b.Float(rho)
    gmean.Assign(gmean.Mul(frho).Add(one.Sub(frho).Mul(xsum)))
    gvar.Assign(gvar.Mul(frho).Add(one.Sub(frho).Mul(xvar)))

    feps := b.Float(eps)
    faccumbeta := b.Float(accumbeta)
    if binary {
        // xhat = shift(x - xsum, 1.0 / sqrt(xvar + eps))
        // outputs = y.reshape(xhat.shape)
        // outputs[] = shift(xhat, gamma) + beta + accumbeta * outputs
        xhat := b.Shift(x.Sub(xsum), one.Div(b.Sqrt(xvar.Add(feps))), true)
        outputs := y.Reshape(xhat.Shape())
        outputs.Assign(b.Shift(xhat, gamma, true).Add(beta).Add(faccumbeta.Mul(outputs)))
    } else {
        // xhat = (x - xsum) / sqrt(xvar + eps)
        // outputs = y.reshape(xhat.shape)
        // outputs[] = xhat * gamma + beta + accumbeta * outputs 
        xhat := x.Sub(xsum).Div(b.Sqrt(xvar.Add(feps)))
        outputs := y.Reshape(xhat.Shape())
        outputs.Assign(xhat.Mul(gamma).Add(beta).Add(faccumbeta.Mul(outputs)))
    }
}

func(b *CpuGenerator) CompoundBpropBn(
        deltaOut backends.Tensor,
        gradGamma backends.Tensor,
        gradBeta backends.Tensor,
        deltaIn backends.Tensor,
        x backends.Tensor,
        xsum backends.Tensor,
        xvar backends.Tensor,
        gamma backends.Tensor,
        eps float64,
        binary bool,
        later backends.BatchNormLayer) {
    var op func(left backends.Value, right backends.Value) backends.Value
    if binary {
        op = func(left backends.Value, right backends.Value) backends.Value {
            return b.Shift(left, right, true)
        }
    } else {
        op = func(left backends.Value, right backends.Value) backends.Value {
            return left.Mul(right)
        }        
    }

    // invV = 1.0 / sqrt(xvar + eps)
    // xhat = op(x - xsum, invV)
    // gradGamma[] = sum(xhat * deltaIn, axis=1)
    // gradBeta[] = sum(delta_in, axis=1)
    // xtmp = (op(xhat, gradGamma) + gradBeta) / float(x.shape[1])
    // deltaOut.reshape(deltaIn.shape)[] = op(op(deltaIn - xtmp, gamma), invV) 
    xshape := x.Shape()
    one := b.Float(1.0)
    feps := b.Float(eps)
    invV := one.Div(b.Sqrt(xvar.Add(feps)))
    xhat := op(x.Sub(xsum), invV)
    gradGamma.Assign(b.Sum(xhat.Mul(deltaIn), 1))
    gradBeta.Assign(b.Sum(deltaIn, 1))
    xtmp := op(xhat, gradGamma).Add(gradBeta).Div(b.Int(xshape[1]))
    deltaOut.Reshape(deltaIn.Shape()).Assign(op(op(deltaIn.Sub(xtmp), gamma), invV))
}

func(b *CpuGenerator) FpropSoftmax(x backends.Value, axis int) backends.Value {
    // reciprocal(sum(exp(x - max(x, axis=axis)), axis=axis)) * exp(x - max(x, axis=axis))
    return b.Reciprocal(b.Sum(b.Exp(x.Sub(b.Max(x, axis))), axis)).Mul(
        b.Exp(x.Sub(b.Max(x, axis))))
}

func(b *CpuGenerator) FpropTransform(
        nglayer backends.Layer, 
        transform backends.Transform, 
        inputs backends.Tensor, 
        outputs backends.Tensor, 
        relu bool) {
    outputs.Assign(transform.Call(inputs))
}

func(b *CpuGenerator) BpropTransform(
        nglayer backends.Layer,
        transform backends.Transform,
        outputs backends.Tensor,
        errors backends.Tensor,
        deltas backends.Tensor,
        relu bool) {
    deltas.Assign(transform.Bprop(outputs).Mul(errors))
}

func(b *CpuGenerator) FpropSkipNode(x backends.Tensor, y backends.Tensor, beta float64) {
    // y[] = y * beta + x
    y.Assign(y.Mul(b.Float(beta)).Add(x))

}

func(b *CpuGenerator) BpropSkipNode(
        errors backends.Tensor, deltas backends.Tensor, alpha float64, beta float64) {
    // deltas[] = deltas * beta + alpha * errors
    deltas.Assign(deltas.Mul(b.Float(beta)).Add(b.Float(alpha).Mul(errors)))
}

// Generator methods

func(b *CpuGenerator) BuildProlog() {
    b.EnterProlog()

    b.WriteLine("")
    b.WriteLine("#include \"%shost.h\"", b.filePrefix)
    b.WriteLine("#include \"%skernels.h\"", b.filePrefix)
    b.WriteLine("")
    b.WriteLine("using namespace arhat;")
    b.WriteLine("using namespace arhat::cpu;")
    b.WriteLine("")

    if len(b.hostNamespace) != 0 {
        b.WriteLine("namespace %s {", b.hostNamespace)
        b.WriteLine("")
    }

    numBuffers := len(b.buffers)
    if numBuffers != 0 {
        b.WriteLine("void *buffers[%d];", numBuffers)
    b.WriteLine("")
    }

    b.DeclareGlobalObjects()

    b.WriteLine("void Prolog() {")
    b.Indent(1)

    b.WriteLine("RngInit();")
    b.WriteLine("")

    rngSeed := b.RngSeed()
    if rngSeed != base.IntNone {
        b.WriteLine("RngSetSeed(%d);", rngSeed)
    }

    for _, buf := range b.buffers {
        base.Assert(buf.offset == 0)
        b.WriteLine("buffers[%d] = CpuMemAlloc(%d);", buf.index, buf.nbytes)
    }

    b.InitializeGlobalObjects()

    b.Indent(-1)
    b.WriteLine("}")
    b.WriteLine("")

    b.ExitProlog()
}

func(b *CpuGenerator) GetData(
        dest string, start string, stop string, x backends.Tensor) string {
    buf := x.(*acc.AccTensor).AccData().(*CpuDeviceAllocation)
    itemSize := x.Dtype().ItemSize()
    return fmt.Sprintf("CpuGetData(%s, %s, %s, %s, %d)", dest, start, stop, buf, itemSize)
}

func(b *CpuGenerator) GetMetricSum(x backends.Tensor, start string, stop string) string {
    buf := x.(*acc.AccTensor).AccData().(*CpuDeviceAllocation)
    bufRef := formatBufferRef(buf, true)
    return fmt.Sprintf("CpuGetFloatSum((float *)%s, %s, %s)", bufRef, start, stop)
}

func(b *CpuGenerator) OutputCode(outDir string) error {
    if b.buildMainCpp {
        if err := b.outputMainCpp(outDir); err != nil {
            return err
        }
    }
    if err := b.outputHostCpp(outDir); err != nil {
        return err
    }
    if err := b.outputKernelsCpp(outDir); err != nil {
        return err
    }
    if err := b.outputHostH(outDir); err != nil {
        return err
    }
    if err := b.outputKernelsH(outDir); err != nil {
        return err
    }
    return nil
}

var mainCppCode = `
#include <cstdio>
#include "%shost.h"

int main() {
    %sMain();
    return 0;
}

`

func(b *CpuGenerator) outputMainCpp(outDir string) error {
    fn := b.filePrefix + "main.cpp"
    fp, err := os.Create(path.Join(outDir, fn))
    if err != nil {
        return err
    } 
    ns := ""
    if len(b.hostNamespace) != 0 {
        ns = b.hostNamespace + "::"
    }
    fmt.Fprintf(fp, mainCppCode, b.filePrefix, ns)
    fp.Close()
    return nil
}

func(b *CpuGenerator) outputHostCpp(outDir string) error {
    fn := b.filePrefix + "host.cpp"
    fp, err := os.Create(path.Join(outDir, fn))
    if err != nil {
        return err
    } 
    code := strings.Join(b.HostChunks(), "")
    fmt.Fprintf(fp, "%s", code)
    if len(b.hostNamespace) != 0 {
        fmt.Fprintf(fp, "} // %s\n", b.hostNamespace)
        fmt.Fprintf(fp, "\n")
    }
    fp.Close()
    return nil
}

func(b *CpuGenerator) outputKernelsCpp(outDir string) error {
    chunks := b.DeviceChunks()
    w := 0
    for n := len(chunks); n > 0; n /= 10 {
        w++
    }
    if w == 0 {
        return nil
    }
    sfxFmt := fmt.Sprintf("%%0%dd", w)

    for index, chunk := range chunks {
        sfx := fmt.Sprintf(sfxFmt, index+1)
        fn := fmt.Sprintf("%skernel%s.cpp", b.filePrefix, sfx)
        fp, err := os.Create(path.Join(outDir, fn))
            if err != nil {
            return err
        } 
        fmt.Fprintf(fp, "\n")
        fmt.Fprintf(fp, "#include \"%skernels.h\"\n", b.filePrefix)
        fmt.Fprintf(fp, "\n")
        fmt.Fprintf(fp, "%s", chunk)
        fmt.Fprintf(fp, "\n")
        fp.Close()
    }
    return nil
}

var hostHCode = `
#pragma once

#include <cstdio>

#include "runtime/arhat.h"
#include "runtime_cpu/arhat.h"

void Main();

`
// end hostHCode

var hostHCodeNs = `
#pragma once

#include <cstdio>

#include "runtime/arhat.h"
#include "runtime_cpu/arhat.h"

namespace %s {

void Main();

} // %s

`
// end hostHCodeNs

func(b *CpuGenerator) outputHostH(outDir string) error {
    fn := b.filePrefix + "host.h"
    fp, err := os.Create(path.Join(outDir, fn))
    if err != nil {
        return err
    } 
    if len(b.hostNamespace) != 0 {
        fmt.Fprintf(fp, hostHCodeNs, b.hostNamespace, b.hostNamespace)
    } else {
        fmt.Fprintf(fp, hostHCode)
    }
    fp.Close()
    return nil
}

func(b *CpuGenerator) outputKernelsH(outDir string) error {
    fn := b.filePrefix + "kernels.h"
    fp, err := os.Create(path.Join(outDir, fn))
    if err != nil {
        return err
    } 
    fmt.Fprintf(fp, "\n")
    fmt.Fprintf(fp, "#pragma once\n")
    fmt.Fprintf(fp, "\n")
    fmt.Fprintf(fp, "#include <cfloat>\n")
    fmt.Fprintf(fp, "#include <cmath>\n")
    fmt.Fprintf(fp, "\n")
    fmt.Fprintf(fp, "#include \"runtime_cpu/arhat_kernels.h\"\n")
    fmt.Fprintf(fp, "\n")

    for _, chunk := range b.DeviceChunks() {
        sub := fmt.Sprintf("void %sKernel_", b.kernelPrefix)
        pos := strings.Index(chunk, sub)
        base.Assert(pos >= 0)
        start := pos
        pos = strings.Index(chunk[pos:], ")")
        base.Assert(pos >= 0)
        sig := chunk[start:start+pos+1]
        fmt.Fprintf(fp, "%s;\n", sig)
        fmt.Fprintf(fp, "\n")
    }
    fp.Close()
    return nil
}

func(b *CpuGenerator) FormatBufferRef(tensor backends.Tensor, paren bool) string {
    accTensor := tensor.(*acc.AccTensor)
    buf := accTensor.AccData()
    cpuBuf := buf.(*CpuDeviceAllocation)
    return formatBufferRef(cpuBuf, paren)
}

// AccGenerator methods

func(b *CpuGenerator) Assign(out *acc.AccTensor, value backends.Value) backends.Value {
    optree := backends.BuildOpTreeNode(backends.Assign, out, value)

    // get post order stack
    stack := optree.Traverse(nil)

    // bypass stage creation
    if acc.IsSimpleStack(stack) {
        return b.CompoundKernel(stack)
    }

    // create stages and evaluate
    stacks := b.SplitToStacks(optree)

    for _, stack = range stacks {
        compoundDot := false
        if len(stack) == 5 {
            if n, ok := stack[3].(*backends.OpTreeNode); ok && n.Op() == backends.Dot {
                compoundDot = true
            }
        }
        if compoundDot {
            // evaluate the simple dot
            b.CompoundDot(
                stack[1].(backends.Tensor), 
                stack[2].(backends.Tensor), 
                stack[0].(backends.Tensor),
                1.0,   // alpha
                0.0,   // beta
                false, // relu
                nil)   // bsum
        } else {
            b.CompoundKernel(stack)
        }       
    }

    // TODO(orig): to be removed, used in partial
    return stacks[len(stacks)-1][0]
}

func(b *CpuGenerator) MemAlloc(nbytes int) acc.DeviceAllocation {
    index := b.nextBufferIndex
    b.nextBufferIndex++
    buf := NewCpuDeviceAllocation(index, nbytes, 0)
    b.buffers = append(b.buffers, buf)
    return buf
}

func(b *CpuGenerator) MemsetD8Async(dest acc.DeviceAllocation, data uint8, count int) {
    buf := formatBufferRef(dest.(*CpuDeviceAllocation), false)
    b.WriteLine("CpuMemsetD8(%s, %du, %d);", buf, data, count)
}

func(b *CpuGenerator) MemsetD16Async(dest acc.DeviceAllocation, data uint16, count int) {
    buf := formatBufferRef(dest.(*CpuDeviceAllocation), false)
    b.WriteLine("CpuMemsetD16(%s, %du, %d);", buf, data, count)
}

func(b *CpuGenerator) MemsetD32Async(dest acc.DeviceAllocation, data uint32, count int) {
    buf := formatBufferRef(dest.(*CpuDeviceAllocation), false)
    b.WriteLine("CpuMemsetD32(%s, %du, %d);", buf, data, count)
}

func(b *CpuGenerator) MemcpyDtodAsync(
        dest acc.DeviceAllocation, src acc.DeviceAllocation, size int) {
    destBuf := formatBufferRef(dest.(*CpuDeviceAllocation), false)
    srcBuf := formatBufferRef(src.(*CpuDeviceAllocation), false)
    b.WriteLine("CpuMemcpy(%s, %s, %d);", destBuf, srcBuf, size)
}

func(b *CpuGenerator) GetInt(src acc.DeviceAllocation, size int) string {
    ref := formatBufferRef(src.(*CpuDeviceAllocation), false)
    return fmt.Sprintf("CpuGetInt(%s, %d)", ref, size)
}

func(b *CpuGenerator) GetFloat(src acc.DeviceAllocation, size int) string {
    ref := formatBufferRef(src.(*CpuDeviceAllocation), false)
    return fmt.Sprintf("CpuGetFloat(%s, %d)", ref, size)
}

// generic compound kernel support

func(b *CpuGenerator) CompoundKernel(args []backends.Value) backends.Value {
    // input: randState is not used for CPU
    // output: threads and reduction not used for CPU
    typeArgs, kernelArgs, out, _, blocks, _ := acc.CompileCompoundKernel(nil, args)

    key := fmt.Sprintf("compound_%s", acc.HashTypeArgs(typeArgs))

    kernel := b.LookupKernel(key)
    if kernel == nil {
        code, name := BuildCompoundKernel(typeArgs)
        id := b.makeKernelId()
        code = strings.Replace(code, name, id, 1)
        code = beautifyKernel(code, true)
        kernel = NewKernel(b, id, code)
        b.RegisterKernel(key, kernel)
        // device code for kernel
        b.EnterKernel()
        b.WriteChunk(code)
        b.ExitKernel()
    }

    kernelArgs = insertBlocks(kernelArgs, blocks)
    kernelSig := makeKernelSig(kernel.code, kernel.name)
    argList := makeKernelArgs(kernelSig, kernelArgs)

    // code for kernel launch
    if len(argList) <= 3 {
        b.WriteLine("%s(%s);", kernel.name, strings.Join(argList, ", "))
    } else {
        b.WriteLine("%s(", kernel.name)
        b.Indent(1)
        n := len(argList)
        for i := 0; i < n; i++ {
            s := argList[i]
            if i != n - 1 {
                s += ","
            } else {
                s += ");"
            }
            b.WriteLine("%s", s)
        }
        b.Indent(-1)
    }

    return out
}

// local helper methods

func(b *CpuGenerator) writeLongStmt(s string, args ...interface{}) {
    stmt := fmt.Sprintf(s, args...)
    lines := strings.Split(stmt, "#")
    b.WriteLine("%s", lines[0])
    b.Indent(1)
    for _, s := range lines[1:] {
        b.WriteLine("%s", s)
    }
    b.Indent(-1)
}

func(b *CpuGenerator) makeKernelId() string {
    index := b.nextKernelIndex
    b.nextKernelIndex++
    return fmt.Sprintf("%sKernel_%d", b.kernelPrefix, index)
}

// local helper functions

func insertBlocks(kernelArgs []acc.KernelArgument, blocks int) []acc.KernelArgument {
    result := make([]acc.KernelArgument, len(kernelArgs)+1)
    result[0] = blocks
    copy(result[1:], kernelArgs)
    return result
}

func makeKernelSig(code string, name string) []string {
    pos := strings.Index(code, name)
    base.Assert(pos >= 0)
    start := pos + len(name)
    pos = strings.Index(code[start:], "(")
    base.Assert(pos >= 0)
    start += pos + 1
    pos = strings.Index(code[start:], ")")
    base.Assert(pos >= 0)
    sig := strings.Split(code[start:start+pos], ",")
    // strip argument names
    mustSkip := func(c byte) bool {
        return ((c >= 'A' && c <= 'Z') ||
            (c >= 'a' && c <= 'z') ||
            (c >= '0' && c <= '9') || c == '_')
    }
    n := len(sig)
    for i := 0; i < n; i++ {
        s := strings.Trim(sig[i], " \n")
        k := len(s) - 1
        for k >= 0 {
            if !mustSkip(s[k]) {
                break
            }
            k--
        }
        base.Assert(k >= 0)
        sig[i] = strings.TrimRight(s[:k+1], " \n")
    }
    return sig
}

func makeKernelArgs(kernelSig []string, kernelArgs []acc.KernelArgument) []string {
    n := len(kernelArgs)
    base.Assert(len(kernelSig) == n)
    result := make([]string, n)
    for i := 0; i < n; i++ {
        switch v := kernelArgs[i].(type) {
        case bool:
            if v {
                result[i] = "true"
            } else {
                result[i] = "false"
            }
        case int:
            result[i] = fmt.Sprintf("%d", v)
        case float64:
            result[i] = formatFloat32(v)
        case string:
            // symbol
            result[i] = v
        case nil:
            // support for nil tensors/buffers
            result[i] = "NULL"
        case *CpuDeviceAllocation:
            result[i] = formatBufferArg(kernelSig[i], v)
        case *acc.AccTensor:
            result[i] = formatBufferArg(kernelSig[i], v.AccData().(*CpuDeviceAllocation))
        default:
            base.ValueError("Argument %d: invalid type: %t", i, kernelArgs[i])
        }
    }
    return result
}

func formatBufferArg(sig string, buf *CpuDeviceAllocation) string {
    if strings.HasPrefix(sig, "const") {
        sig = strings.TrimLeft(sig[5:], " \n")
    }
    return fmt.Sprintf("(%s)%s", sig, formatBufferRef(buf, true))
}

func formatFloat32(x float64) string {
    return generators.FormatFloat32(x)
}

func formatBufferRef(buf *CpuDeviceAllocation, paren bool) string {
    if buf.offset != 0 {
        if (paren) {
            return fmt.Sprintf("((uint8_t *)buffers[%d]+%d)", buf.index, buf.offset)
        } else {
            return fmt.Sprintf("(uint8_t *)buffers[%d]+%d", buf.index, buf.offset)
        }
    } else {
        return fmt.Sprintf("buffers[%d]", buf.index)
    }
}

//
//    Minimum beautify procedure for legacy kernels
//

func beautifyKernel(code string, autoIndent bool) string {
    var result []string
    emptyLine := false
    indent := 0
    lines := strings.Split(code, "\n")
    for _, line := range lines {
        content := strings.Trim(line, " \t")
        if len(content) == 0 {
            if !emptyLine {
                result = append(result, "")
                emptyLine = true
            }
        } else {
            if autoIndent {
                lb := strings.Contains(line, "{")
                rb := strings.Contains(line, "}")
                if rb && !lb {
                    indent--
                }
                if indent != 0 {
                    // retain layout at top level
                    line = strings.Repeat(" ", 4*indent) + content
                }
                if lb && !rb {
                    indent++
                }
            }
            result = append(result, line)
            emptyLine = false
        }
    }
    if !emptyLine {
        // code doesn't end with '\n'
        result = append(result, "")
    }
    result = append(result, "")
    return strings.Join(result, "\n")
}

