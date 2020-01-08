//
// Copyright 2019-2020 FRAGATA COMPUTER SYSTEMS AG
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

package main

import (
    "fmt"
    "fragata/arhat/backends"
    "fragata/arhat/base"
    "fragata/arhat/generators/cuda"
    "os"
    "strings"
)

//
//    Configurable parameters
//

var (
    computeCapability = [2]int{6, 0}
)

var (
    // number of full iterations
    loops = 10
    // show bechmark details for each layer
    layerBench = false
    // show layer stats after each operation
    printStats = false
    // run network with all zeros to see speed difference
    zeros = false
    // print more stuff
    verbose = true
)

//
//    Conf
//

type Conf map[string]interface{}

func(c Conf) Clone() Conf {
    r := make(Conf)
    for k, v := range c {
        r[k] = v
    }
    return r
}

func(c Conf) Update(other Conf) {
    for k, v := range other {
        c[k] = v
    }
}

func(c Conf) GetBool(key string) bool {
    val, ok := c[key]
    if !ok {
        return false
    }
    return val.(bool)
}

func(c Conf) GetInt(key string) int {
    val, ok := c[key]
    if !ok {
        return base.IntNone
    }
    return val.(int)
}

func(c Conf) GetString(key string) string {
    val, ok := c[key]
    if !ok {
        return ""
    }
    return val.(string)
}

//
//    Common configuration fragments
//

// common convolutional layer settings

var (
    conv11    = Conf{"R": 11, "S": 11, "pad_h": 2, "pad_w": 2, "str_h": 4, "str_w": 4}
    conv11p0  = Conf{"R": 11, "S": 11, "pad_h": 0, "pad_w": 0, "str_h": 4, "str_w": 4}
    conv7     = Conf{"R": 7,  "S": 7,  "pad_h": 3, "pad_w": 3, "str_h": 2, "str_w": 2}
    conv5     = Conf{"R": 5,  "S": 5,  "pad_h": 2, "pad_w": 2}
    conv5p0   = Conf{"R": 5,  "S": 5,  "pad_h": 0, "pad_w": 0}
    conv3     = Conf{"R": 3,  "S": 3,  "pad_h": 1, "pad_w": 1}
    conv2     = Conf{"R": 2,  "S": 2,  "pad_h": 0, "pad_w": 0, "str_h": 2, "str_w": 2}
    conv1     = Conf{"R": 1,  "S": 1,  "pad_h": 0, "pad_w": 0}
)

// traditional pooling

var (
    pool2s2p0 = Conf{"R": 2, "S": 2}
    pool3s2p0 = Conf{"R": 3, "S": 3, "str_h": 2, "str_w": 2}
    pool3s2p1 = Conf{"R": 3, "S": 3, "str_h": 2, "str_w": 2, "pad_h": 1, "pad_w": 1}
    pool3s1p1 = Conf{"R": 3, "S": 3, "str_h": 1, "str_w": 1, "pad_h": 1, "pad_w": 1}
    pool7s1p0 = Conf{"R": 7, "S": 7, "str_h": 1, "str_w": 1}
)

// maxout pooling

var (
    pool1j2 = Conf{"op": "max", "J": 2} // maxout in the fc layers
    pool2j2 = Conf{"op": "max", "J": 2, "R": 2, "S": 2}
    pool3j2 = Conf{"op": "max", "J": 2, "R": 3, "S": 3}
)

//
//    Layer constructor (originally in backends/layer_gpu.py)
//

func CreateLayer(
        lib *cuda.CudaGenerator, 
        conf Conf, 
        prevLayer cuda.Layer, 
        dtype base.Dtype) cuda.Layer {
    config := conf.Clone()
    layerType := config.GetString("layer")
    delete(config, "layer")

    // merge dtype specific settings
    config["dtype"] = dtype

    // merge shared params
    common := config["common"]
    delete(config, "common")
    if common != nil {
        config.Update(common.(Conf))
    }

    // Propagate the fixed and calculated dimensions
    if prevLayer != nil {
        config["N"] = prevLayer.N()
        _, isPrevFullLayer := prevLayer.(*cuda.FullLayer)

        switch {
        case layerType == "FullLayer":
            config["nIn"] = prevLayer.NOut()

        case layerType == "PoolLayer" && isPrevFullLayer:
            config["C"] = prevLayer.NOut()

        case layerType == "BatchNorm" && isPrevFullLayer:
            config["nIn"] = prevLayer.NOut()

        default:
            config["C"] = prevLayer.K()
            config["D"] = prevLayer.M()
            config["H"] = prevLayer.P()
            config["W"] = prevLayer.Q()

            if layerType == "Inception" {
                partitions := config["partitions"].([][]Conf)
                configK := 0
                var configPartitions [][]cuda.Layer

                for _, part := range partitions {
                    var layerSequence []cuda.Layer
                    last := prevLayer
                    for _, layerConf := range part {
                        last = CreateLayer(lib, layerConf, last, dtype)
                        layerSequence = append(layerSequence, last)
                    }

                    configPartitions = append(configPartitions, layerSequence)
                    configK += last.K()
                    if _, ok := config["P"]; ok {
                        base.Assert(
                            config.GetInt("P") == last.P() && 
                            config.GetInt("Q") == last.Q())
                    } else {
                        config["M"] = last.M()
                        config["P"] = last.P()
                        config["Q"] = last.Q()
                    }
                }

                config["K"] = configK
                config["partitions"] = configPartitions
            }
        }
    }
              
    // Instantiate the layer
    switch layerType {
    case "DataLayer":
        return CreateDataLayer(lib, config)
    case "ConvLayer":
        return CreateConvLayer(lib, config)
    case "PoolLayer":
        return CreatePoolLayer(lib, config)
    case "FullLayer":
        return CreateFullLayer(lib, config)
    case "Inception":
        return CreateInception(lib, config)
    case "BatchNorm":
        return CreateBatchNorm(lib, config)
    default:
        base.ValueError("Invalid layer type:  %s", layerType)
    }

    return nil
}

func CreateDataLayer(lib *cuda.CudaGenerator, conf Conf) *cuda.DataLayer {
    dtype := base.DtypeNone
    var a cuda.DataParams
    a.Init()

    for k, v := range conf {
        switch k {
        case "dtype":
            dtype = v.(base.Dtype)
        case "N":
            a.N = v.(int)
        case "C":
            a.C = v.(int)
        case "D":
            a.D = v.(int)
        case "H":
            a.H = v.(int)
        case "W":
            a.W = v.(int)
        default:
            base.ValueError("Invalid DataLayer parameter: %s", k)
        }
    }

    base.Assert(dtype != base.DtypeNone)
    return cuda.NewDataLayer(lib, dtype, &a)
}

func CreateConvLayer(lib *cuda.CudaGenerator, conf Conf) *cuda.ConvLayer {
    dtype := base.DtypeNone
    var a backends.ConvParams
    a.Init()

    relu := false
    bsum := false

    for k, v := range conf {
        switch k {
        case "dtype":
            dtype = v.(base.Dtype)
        case "N":
            a.N = v.(int)
        case "C":
            a.C = v.(int)
        case "K":
            a.K = v.(int)
        case "D":
            a.D = v.(int)
        case "H":
            a.H = v.(int)
        case "W":
            a.W = v.(int)
        case "T":
            a.T = v.(int)
        case "R":
            a.R = v.(int)
        case "S":
            a.S = v.(int)
        case "pad_d":
            a.PadD = v.(int)
        case "pad_h":
            a.PadH = v.(int)
        case "pad_w":
            a.PadW = v.(int)
        case "str_d":
            a.StrD = v.(int)
        case "str_h":
            a.StrH = v.(int)
        case "str_w":
            a.StrW = v.(int)
        case "dil_d":
            a.DilD = v.(int)
        case "dil_h":
            a.DilH = v.(int)
        case "dil_w":
            a.DilW = v.(int)
        case "relu":
            relu = v.(bool)        
        case "bsum":
            bsum = v.(bool)
        default:
            base.ValueError("Invalid ConvLayer parameter: %s", k)
        }
    }

    base.Assert(dtype != base.DtypeNone)
    layer := cuda.NewConvLayer(lib, dtype, &a, lib)
    layer.SetRelu(relu)
    layer.SetBsum(bsum)
    return layer
}

func CreatePoolLayer(lib *cuda.CudaGenerator, conf Conf) *cuda.PoolLayer {
    dtype := base.DtypeNone

    opStr := conf.GetString("op")
    delete(conf, "op")
    var op backends.PoolOp
    switch opStr {
    case "avg":
        op = backends.PoolOpAvg
    case "max":
        op = backends.PoolOpMax
    default:
        base.ValueError("Invalid pool op: %s\n", opStr)
    }

    var a backends.PoolParams
    a.Init(op)

    for k, v := range conf {
        switch k {
        case "dtype":
            dtype = v.(base.Dtype)
        case "N":
            a.N = v.(int)
        case "C":
            a.C = v.(int)
        case "D":
            a.D = v.(int)
        case "H":
            a.H = v.(int)
        case "W":
            a.W = v.(int)
        case "J":
            a.J = v.(int)
        case "T":
            a.T = v.(int)
        case "R":
            a.R = v.(int)
        case "S":
            a.S = v.(int)
        case "pad_c":
            a.PadC = v.(int)
        case "pad_d":
            a.PadD = v.(int)
        case "pad_h":
            a.PadH = v.(int)
        case "pad_w":
            a.PadW = v.(int)
        case "str_c":
            a.StrC = v.(int)
        case "str_d":
            a.StrD = v.(int)
        case "str_h":
            a.StrH = v.(int)
        case "str_w":
            a.StrW = v.(int)

        default:
            base.ValueError("Invalid PoolLayer parameter: %s", k)
        }
    }
    base.Assert(dtype != base.DtypeNone)
    return cuda.NewPoolLayer(lib, dtype, &a)
}

func CreateFullLayer(lib *cuda.CudaGenerator, conf Conf) *cuda.FullLayer {
    dtype := base.DtypeNone

    var a cuda.FullParams
    a.Init()

    for k, v := range conf {
        switch k {
        case "dtype":
            dtype = v.(base.Dtype)
        case "N":
            a.N = v.(int)
        case "nIn":
            a.NIn = v.(int)
        case "nOut":
            a.NOut = v.(int)
        case "relu":
            a.Relu = v.(bool)
        default:
            base.ValueError("Invalid FullLayer parameter: %s", k)
        }
    }

    base.Assert(dtype != base.DtypeNone)
    return cuda.NewFullLayer(lib, dtype, &a)
}

func CreateInception(lib *cuda.CudaGenerator, conf Conf) *cuda.InceptionLayer {
    dtype := base.DtypeNone
    var partitions [][]cuda.Layer

    var a cuda.InceptionParams
    a.Init()

    for k, v := range conf {
        switch k {
        case "dtype":
            dtype = v.(base.Dtype)
        case "partitions":
            partitions = v.([][]cuda.Layer)
        case "N":
            a.N = v.(int)
        case "C":
            a.C = v.(int)
        case "K":
            a.K = v.(int)
        case "D":
            a.D = v.(int)
        case "H":
            a.H = v.(int)
        case "W":
            a.W = v.(int)
        case "M":
            a.M = v.(int)
        case "P":
            a.P = v.(int)
        case "Q":
            a.Q = v.(int)
        default:
            base.ValueError("Invalid Inception parameter: %s", k)
        }
    }

    base.Assert(dtype != base.DtypeNone)
    base.Assert(partitions != nil)
    return cuda.NewInceptionLayer(lib, dtype, partitions, &a)
}

func CreateBatchNorm(lib *cuda.CudaGenerator, conf Conf) *cuda.BatchNormLayer {
    dtype := base.DtypeNone

    var a cuda.BatchNormParams
    a.Init()

    for k, v := range conf {
        switch k {
        case "dtype":
            dtype = v.(base.Dtype)
        case "N":
            a.N = v.(int)
        case "C":
            a.C = v.(int)
        case "D":
            a.D = v.(int)
        case "H":
            a.H = v.(int)
        case "W":
            a.W = v.(int)
        case "nIn":
            a.NIn = v.(int)
        case "rho":
            a.Rho = v.(float64)
        case "eps":
            a.Eps = v.(float64)
        case "relu":
            a.Relu = v.(bool)
        case "bsum":
            a.Bsum = v.(bool)
        default:
            base.ValueError("Invalid BatchNorm parameter: %s", k)
        }
    }

    base.Assert(dtype != base.DtypeNone)
    return cuda.NewBatchNormLayer(lib, dtype, &a)
}

//
//    Inception construction helpers
//

func Inception1(c00 int, c10 int, c11 int, c20 int, c21 int, c30 int) Conf {
    return Conf{
        "layer": "Inception", 
        "partitions": [][]Conf{
            []Conf{
                Conf{"layer": "ConvLayer", "common": conv1, "relu": true, "K": c00},
            },
            []Conf{
                Conf{"layer": "ConvLayer", "common": conv1, "relu": true, "K": c10},
                Conf{"layer": "ConvLayer", "common": conv3, "relu": true, "K": c11},
            },
            []Conf{
                Conf{"layer": "ConvLayer", "common": conv1, "relu": true, "K": c20},
                Conf{"layer": "ConvLayer", "common": conv5, "relu": true, "K": c21},
            },
            []Conf{
                Conf{"layer": "PoolLayer", "common": pool3s1p1, "op": "max"},
                Conf{"layer": "ConvLayer", "common": conv1, "relu": true, "K": c30},
            },
        },
    }
}

func Inception1BN(c00 int, c10 int, c11 int, c20 int, c21 int, c30 int) Conf {
    return Conf{
        "layer": "Inception", 
        "partitions": [][]Conf{
            []Conf{
                Conf{"layer": "ConvLayer", "common": conv1, "K": c00, "bsum": true},
                Conf{"layer": "BatchNorm", "relu": true, "bsum": true},
            },
            []Conf{
                Conf{"layer": "ConvLayer", "common": conv1, "K": c10, "bsum": true},
                Conf{"layer": "BatchNorm", "relu": true, "bsum": true},
                Conf{"layer": "ConvLayer", "common": conv3, "K": c11, "bsum": true},
                Conf{"layer": "BatchNorm", "relu": true, "bsum": true},
            },
            []Conf{
                Conf{"layer": "ConvLayer", "common": conv1, "K": c20, "bsum": true},
                Conf{"layer": "BatchNorm", "relu": true, "bsum": true},
                Conf{"layer": "ConvLayer", "common": conv5, "K": c21, "bsum": true},
                Conf{"layer": "BatchNorm", "relu": true, "bsum": true},
            },
            []Conf{
                Conf{"layer": "PoolLayer", "common": pool3s1p1, "op": "max"},
                Conf{"layer": "ConvLayer", "common": conv1, "K": c30, "bsum": true},
                Conf{"layer": "BatchNorm", "relu": true, "bsum": true},
            },
        },
    }
}

func Inception2(c00 int, c10 int, c11 int, c20 int, c21 int, c30 string, c31 int) Conf {
    layer := Conf{"layer": "Inception"}
    var partitions [][]Conf

    if c00 != 0 {
        partitions = append(partitions, []Conf{
            Conf{"layer": "ConvLayer", "common": conv1, "relu": true, "K": c00},
        })
    }

    partitions = append(partitions, []Conf{
        Conf{"layer": "ConvLayer", "common": conv1, "relu": true, "K": c10},
        Conf{"layer": "ConvLayer", "common": conv3, "relu": true, "K": c11},
    })
    partitions = append(partitions, []Conf{
        Conf{"layer": "ConvLayer", "common": conv1, "relu": true, "K": c20},
        Conf{"layer": "ConvLayer", "common": conv3, "relu": true, "K": c21},
        Conf{"layer": "ConvLayer", "common": conv3, "relu": true, "K": c21},
    })

    if c31 != 0 {
        partitions = append(partitions, []Conf{
            Conf{"layer": "PoolLayer", "common": pool3s1p1, "op": c30},
            Conf{"layer": "ConvLayer", "common": conv1, "relu": true, "K": c31},
        })
    } else {
        partitions = append(partitions, []Conf{
            Conf{"layer": "PoolLayer", "common": pool3s1p1, "op": c30},
        })
    }

    layer["partitions"] = partitions
    return layer
}

func Inception2BN(c00 int, c10 int, c11 int, c20 int, c21 int, c30 string, c31 int) Conf {
    layer := Conf{"layer": "Inception"}
    var partitions [][]Conf

    if c00 != 0 {
        partitions = append(partitions, []Conf{
            Conf{"layer": "ConvLayer", "common": conv1, "K": c00, "bsum": true},
            Conf{"layer": "BatchNorm", "relu": true, "bsum": true},
        })
    }

    partitions = append(partitions, []Conf{
        Conf{"layer": "ConvLayer", "common": conv1, "K": c10, "bsum": true},
        Conf{"layer": "BatchNorm", "relu": true, "bsum": true},
        Conf{"layer": "ConvLayer", "common": conv3, "K": c11, "bsum": true},
        Conf{"layer": "BatchNorm", "relu": true, "bsum": true},
    })
    partitions = append(partitions, []Conf{
        Conf{"layer": "ConvLayer", "common": conv1, "K": c20, "bsum": true},
        Conf{"layer": "BatchNorm", "relu": true, "bsum": true},
        Conf{"layer": "ConvLayer", "common": conv3, "K": c21, "bsum": true},
        Conf{"layer": "BatchNorm", "relu": true, "bsum": true},
        Conf{"layer": "ConvLayer", "common": conv3, "K": c21, "bsum": true},
        Conf{"layer": "BatchNorm", "relu": true, "bsum": true },
    })

    if c31 != 0 {
        partitions = append(partitions, []Conf{
            Conf{"layer": "PoolLayer", "common": pool3s1p1, "op": c30},
            Conf{"layer": "ConvLayer", "common": conv1, "K": c31, "bsum": true},
            Conf{"layer": "BatchNorm", "relu": true, "bsum": true},
        })
    } else {
        partitions = append(partitions, []Conf{
            Conf{"layer": "PoolLayer", "common": pool3s1p1, "op": c30},
        })
    }

    layer["partitions"] = partitions
    return layer
}

//
//    Networks
//

var networks = map[string][]Conf{
    "Alexnet": []Conf{
        Conf{"warmup": 1},
        Conf{"layer": "DataLayer", "N": 128, "C": 3, "H": 224, "W": 224},
        Conf{"layer": "ConvLayer", "common": conv11, "relu": true,"K": 64},
        Conf{"layer": "PoolLayer", "common": pool3s2p0, "op": "max"},
        Conf{"layer": "ConvLayer", "common": conv5, "relu": true, "K": 192},
        Conf{"layer": "PoolLayer", "common": pool3s2p0, "op": "max"},
        Conf{"layer": "ConvLayer", "common": conv3, "relu": true, "K": 384},
        Conf{"layer": "ConvLayer", "common": conv3, "relu": true, "K": 256},
        Conf{"layer": "ConvLayer", "common": conv3, "relu": true, "K": 256},
        Conf{"layer": "PoolLayer", "common": pool3s2p0, "op": "max"},
        Conf{"layer": "FullLayer", "nOut": 4096, "relu": true},
        Conf{"layer": "FullLayer", "nOut": 4096, "relu": true},
        Conf{"layer": "FullLayer", "nOut": 1000, "relu": true},
    },
    "AlexnetBN": []Conf{
        Conf{"warmup": 1},
        Conf{"layer": "DataLayer", "N": 128, "C": 3, "H": 224, "W": 224},
        Conf{"layer": "ConvLayer", "common": conv11,"K": 64, "bsum": true},
        Conf{"layer": "BatchNorm", "relu": true, "bsum": true},
        Conf{"layer": "PoolLayer", "common": pool3s2p0, "op": "max"},
        Conf{"layer": "ConvLayer", "common": conv5, "K": 192, "bsum": true},
        Conf{"layer": "BatchNorm", "relu": true, "bsum": true},
        Conf{"layer": "PoolLayer", "common": pool3s2p0, "op": "max"},
        Conf{"layer": "ConvLayer", "common": conv3, "K": 384, "bsum": true},
        Conf{"layer": "BatchNorm", "relu": true, "bsum": true},
        Conf{"layer": "ConvLayer", "common": conv3, "K": 256, "bsum": true},
        Conf{"layer": "BatchNorm", "relu": true, "bsum": true},
        Conf{"layer": "ConvLayer", "common": conv3, "K": 256, "bsum": true},
        Conf{"layer": "BatchNorm", "relu": true, "bsum": true},
        Conf{"layer": "PoolLayer", "common": pool3s2p0, "op": "max"},
        Conf{"layer": "FullLayer", "nOut": 4096},
        Conf{"layer": "BatchNorm", "relu": true},
        Conf{"layer": "FullLayer", "nOut": 4096},
        Conf{"layer": "BatchNorm", "relu": true},
        Conf{"layer": "FullLayer", "relu": true, "nOut": 1000},
    },
    "Overfeat": []Conf{
        Conf{"warmup": 1},
        Conf{"layer": "DataLayer", "N": 128, "C": 3, "H": 231, "W": 231},
        Conf{"layer": "ConvLayer", "common": conv11p0, "relu": true,"K": 96},
        Conf{"layer": "PoolLayer", "common": pool2s2p0, "op": "max"},
        Conf{"layer": "ConvLayer", "common": conv5p0, "relu": true, "K": 256},
        Conf{"layer": "PoolLayer", "common": pool2s2p0, "op": "max"},
        Conf{"layer": "ConvLayer", "common": conv3, "relu": true,   "K": 512},
        Conf{"layer": "ConvLayer", "common": conv3, "relu": true,   "K": 1024},
        Conf{"layer": "ConvLayer", "common": conv3, "relu": true,   "K": 1024},
        Conf{"layer": "PoolLayer", "common": pool2s2p0, "op": "max"},
        Conf{"layer": "FullLayer", "nOut": 3072, "relu": true},
        Conf{"layer": "FullLayer", "nOut": 4096, "relu": true},
        Conf{"layer": "FullLayer", "nOut": 1000, "relu": true},
    },
    "OverfeatBN": []Conf{
        Conf{"warmup": 1},
        Conf{"layer": "DataLayer", "N": 128, "C": 3, "H": 231, "W": 231},
        Conf{"layer": "ConvLayer", "common": conv11p0,"K": 96, "bsum": true},
        Conf{"layer": "BatchNorm", "relu": true, "bsum": true},
        Conf{"layer": "PoolLayer", "common": pool2s2p0, "op": "max"},
        Conf{"layer": "ConvLayer", "common": conv5p0, "K": 256, "bsum": true},
        Conf{"layer": "BatchNorm", "relu": true, "bsum": true},
        Conf{"layer": "PoolLayer", "common": pool2s2p0, "op": "max"},
        Conf{"layer": "ConvLayer", "common": conv3,   "K": 512, "bsum": true},
        Conf{"layer": "BatchNorm", "relu": true, "bsum": true},
        Conf{"layer": "ConvLayer", "common": conv3,   "K": 1024, "bsum": true},
        Conf{"layer": "BatchNorm", "relu": true, "bsum": true},
        Conf{"layer": "ConvLayer", "common": conv3,   "K": 1024, "bsum": true},
        Conf{"layer": "BatchNorm", "relu": true, "bsum": true},
        Conf{"layer": "PoolLayer", "common": pool2s2p0, "op": "max"},
        Conf{"layer": "FullLayer", "nOut": 3072},
        Conf{"layer": "BatchNorm", "relu": true},
        Conf{"layer": "FullLayer", "nOut": 4096},
        Conf{"layer": "BatchNorm", "relu": true},
        Conf{"layer": "FullLayer", "relu": true, "nOut": 1000},
    },
    // See http://arxiv.org/pdf/1409.1556.pdf for variations
    "VGG": []Conf{
        Conf{"warmup": 1},
        Conf{"layer": "DataLayer", "N": 64, "C": 3, "H": 224, "W": 224},
        Conf{"layer": "ConvLayer", "common": conv3, "relu": true, "K": 64},
        Conf{"layer": "PoolLayer", "common": pool2s2p0, "op": "max"},
        Conf{"layer": "ConvLayer", "common": conv3, "relu": true, "K": 128},
        Conf{"layer": "PoolLayer", "common": pool2s2p0, "op": "max"},
        Conf{"layer": "ConvLayer", "common": conv3, "relu": true, "K": 256},
        Conf{"layer": "ConvLayer", "common": conv3, "relu": true, "K": 256},
        Conf{"layer": "PoolLayer", "common": pool2s2p0, "op": "max"},
        Conf{"layer": "ConvLayer", "common": conv3, "relu": true, "K": 512},
        Conf{"layer": "ConvLayer", "common": conv3, "relu": true, "K": 512},
        Conf{"layer": "PoolLayer", "common": pool2s2p0, "op": "max"},
        Conf{"layer": "ConvLayer", "common": conv3, "relu": true, "K": 512},
        Conf{"layer": "ConvLayer", "common": conv3, "relu": true, "K": 512},
        Conf{"layer": "PoolLayer", "common": pool2s2p0, "op": "max"},
        Conf{"layer": "FullLayer", "nOut": 4096, "relu": true},
        Conf{"layer": "FullLayer", "nOut": 4096, "relu": true},
        Conf{"layer": "FullLayer", "nOut": 1000, "relu": true},
    },
    // See http://arxiv.org/pdf/1409.1556.pdf for variations
    "VGG_BN": []Conf{
        Conf{"warmup": 1},
        Conf{"layer": "DataLayer", "N": 64, "C": 3, "H": 224, "W": 224},
        Conf{"layer": "ConvLayer", "common": conv3, "K": 64, "bsum": true},
        Conf{"layer": "BatchNorm", "relu": true, "bsum": true},
        Conf{"layer": "PoolLayer", "common": pool2s2p0, "op": "max"},
        Conf{"layer": "ConvLayer", "common": conv3, "K": 128, "bsum": true},
        Conf{"layer": "BatchNorm", "relu": true, "bsum": true},
        Conf{"layer": "PoolLayer", "common": pool2s2p0, "op": "max"},
        Conf{"layer": "ConvLayer", "common": conv3, "K": 256, "bsum": true},
        Conf{"layer": "BatchNorm", "relu": true, "bsum": true},
        Conf{"layer": "ConvLayer", "common": conv3, "K": 256, "bsum": true},
        Conf{"layer": "BatchNorm", "relu": true, "bsum": true},
        Conf{"layer": "PoolLayer", "common": pool2s2p0, "op": "max"},
        Conf{"layer": "ConvLayer", "common": conv3, "K": 512, "bsum": true},
        Conf{"layer": "BatchNorm", "relu": true, "bsum": true},
        Conf{"layer": "ConvLayer", "common": conv3, "K": 512, "bsum": true},
        Conf{"layer": "BatchNorm", "relu": true, "bsum": true},
        Conf{"layer": "PoolLayer", "common": pool2s2p0, "op": "max"},
        Conf{"layer": "ConvLayer", "common": conv3, "K": 512, "bsum": true},
        Conf{"layer": "BatchNorm", "relu": true, "bsum": true},
        Conf{"layer": "ConvLayer", "common": conv3, "K": 512, "bsum": true},
        Conf{"layer": "BatchNorm", "relu": true, "bsum": true},
        Conf{"layer": "PoolLayer", "common": pool2s2p0, "op": "max"},
        Conf{"layer": "FullLayer", "nOut": 4096},
        Conf{"layer": "BatchNorm", "relu": true},
        Conf{"layer": "FullLayer", "nOut": 4096},
        Conf{"layer": "BatchNorm", "relu": true},
        Conf{"layer": "FullLayer", "nOut": 1000, "relu": true},
    },
    // Here is the biggest VGG model (19 layers)
    "VGG_E": []Conf{
        Conf{"warmup": 1},
        Conf{"layer": "DataLayer", "N": 64, "C": 3, "H": 224, "W": 224},
        Conf{"layer": "ConvLayer", "common": conv3, "relu": true, "K": 64},
        Conf{"layer": "ConvLayer", "common": conv3, "relu": true, "K": 64},
        Conf{"layer": "PoolLayer", "common": pool2s2p0, "op": "max"},
        Conf{"layer": "ConvLayer", "common": conv3, "relu": true, "K": 128},
        Conf{"layer": "ConvLayer", "common": conv3, "relu": true, "K": 128},
        Conf{"layer": "PoolLayer", "common": pool2s2p0, "op": "max"},
        Conf{"layer": "ConvLayer", "common": conv3, "relu": true, "K": 256},
        Conf{"layer": "ConvLayer", "common": conv3, "relu": true, "K": 256},
        Conf{"layer": "ConvLayer", "common": conv3, "relu": true, "K": 256},
        Conf{"layer": "ConvLayer", "common": conv3, "relu": true, "K": 256},
        Conf{"layer": "PoolLayer", "common": pool2s2p0, "op": "max"},
        Conf{"layer": "ConvLayer", "common": conv3, "relu": true, "K": 512},
        Conf{"layer": "ConvLayer", "common": conv3, "relu": true, "K": 512},
        Conf{"layer": "ConvLayer", "common": conv3, "relu": true, "K": 512},
        Conf{"layer": "ConvLayer", "common": conv3, "relu": true, "K": 512},
        Conf{"layer": "PoolLayer", "common": pool2s2p0, "op": "max"},
        Conf{"layer": "ConvLayer", "common": conv3, "relu": true, "K": 512},
        Conf{"layer": "ConvLayer", "common": conv3, "relu": true, "K": 512},
        Conf{"layer": "ConvLayer", "common": conv3, "relu": true, "K": 512},
        Conf{"layer": "ConvLayer", "common": conv3, "relu": true, "K": 512},
        Conf{"layer": "PoolLayer", "common": pool2s2p0, "op": "max"},
        Conf{"layer": "FullLayer", "nOut": 4096, "relu": true},
        Conf{"layer": "FullLayer", "nOut": 4096, "relu": true},
        Conf{"layer": "FullLayer", "nOut": 1000, "relu": true},
    },
    // http://arxiv.org/abs/1409.4842
    "GoogLeNet1": []Conf{
        Conf{"warmup": 1},
        Conf{"layer": "DataLayer", "N": 128, "C": 3, "H": 224, "W": 224},
        Conf{"layer": "ConvLayer", "common": conv7, "relu": true, "K": 64},
        Conf{"layer": "PoolLayer", "common": pool3s2p1, "op": "max"},
        Conf{"layer": "ConvLayer", "common": conv1, "relu": true, "K": 64},
        Conf{"layer": "ConvLayer", "common": conv3, "relu": true, "K": 192},
        Conf{"layer": "PoolLayer", "common": pool3s2p1, "op": "max"},
        Inception1(64, 96, 128, 16, 32, 32),
        Inception1(128, 128, 192, 32, 96, 64),
        Conf{"layer": "PoolLayer", "common": pool3s2p1, "op": "max"},
        Inception1(192, 96, 208, 16, 48, 64),
        Inception1(160, 112, 224, 24, 64, 64),
        Inception1(128, 128, 256, 24, 64, 64),
        Inception1(112, 144, 288, 32, 64, 64),
        Inception1(256, 160, 320, 32, 128, 128),
        Conf{"layer": "PoolLayer", "common": pool3s2p1, "op": "max"},
        Inception1(256, 160, 320, 32, 128, 128),
        Inception1(384, 192, 384, 48, 128, 128),
        Conf{"layer": "PoolLayer", "common": pool7s1p0, "op": "avg"},
        Conf{"layer": "FullLayer", "nOut": 1000, "relu": true},
    },
    // http://arxiv.org/abs/1409.4842
    "GoogLeNet1BN": []Conf{
        Conf{"warmup":  1},
        Conf{"layer":  "DataLayer", "N":  128, "C":  3, "H":  224, "W":  224},
        Conf{"layer":  "ConvLayer", "common":  conv7, "K":  64, "bsum":  true},
        Conf{"layer": "BatchNorm", "relu": true, "bsum": true},
        Conf{"layer": "PoolLayer", "common": pool3s2p1, "op": "max"},
        Conf{"layer": "ConvLayer", "common": conv1, "K": 64, "bsum": true},
        Conf{"layer": "BatchNorm", "relu": true, "bsum": true},
        Conf{"layer": "ConvLayer", "common": conv3, "K": 192, "bsum": true},
        Conf{"layer": "BatchNorm", "relu": true, "bsum": true},
        Conf{"layer": "PoolLayer", "common": pool3s2p1, "op": "max"},
        Inception1BN(64, 96, 128, 16, 32, 32),
        Inception1BN(128, 128, 192, 32, 96, 64),
        Conf{"layer": "PoolLayer", "common": pool3s2p1, "op": "max"},
        Inception1BN(192, 96, 208, 16, 48, 64),
        Inception1BN(160, 112, 224, 24, 64, 64),
        Inception1BN(128, 128, 256, 24, 64, 64),
        Inception1BN(112, 144, 288, 32, 64, 64),
        Inception1BN(256, 160, 320, 32, 128, 128),
        Conf{"layer": "PoolLayer", "common": pool3s2p1, "op": "max"},
        Inception1BN(256, 160, 320, 32, 128, 128),
        Inception1BN(384, 192, 384, 48, 128, 128),
        Conf{"layer": "PoolLayer", "common": pool7s1p0, "op": "avg"},
        Conf{"layer": "FullLayer", "nOut": 1000, "relu": true},
    },
    // adapted from: https://github.com/soumith/kaggle_retinopathy_starter.torch/blob/master/models/googlenet_cudnn.lua
    "GoogLeNet2": []Conf{
        Conf{"warmup": 1},
        Conf{"layer": "DataLayer", "N": 128, "C": 3, "H": 224, "W": 224},
        Conf{"layer": "ConvLayer", "common": conv7, "relu": true, "K": 64},
        Conf{"layer": "PoolLayer", "common": pool3s2p1, "op": "max"},
        Conf{"layer": "ConvLayer", "common": conv1, "relu": true, "K": 64},
        Conf{"layer": "ConvLayer", "common": conv3, "relu": true, "K": 192},
        Conf{"layer": "PoolLayer", "common": pool3s2p1, "op": "max"},
        Inception2(64, 64, 64, 64, 96, "avg", 32),
        Inception2(64, 64, 96, 64, 96, "avg", 64),
        Inception2(0, 128, 160, 64, 96, "max", 0),
        Conf{"layer": "ConvLayer", "common": conv2, "relu": true, "K": 576},
        Inception2(224, 64, 96, 96, 128, "avg", 128),
        Inception2(192, 96, 128, 96, 128, "avg" ,128),
        Inception2(160, 128, 160, 128, 160, "avg", 96),
        Inception2(96, 128, 192, 160, 192, "avg", 96),
        Inception2(0, 128, 192, 192, 256, "max", 0),
        Conf{"layer": "ConvLayer", "common": conv2, "relu": true, "K": 1024},
        Inception2(352, 192, 320, 160, 224, "avg", 128),
        Inception2(352, 192, 320, 192, 224, "max", 128),
        Conf{"layer": "PoolLayer", "common": pool7s1p0, "op": "avg"},
        Conf{"layer": "FullLayer", "nOut": 1000, "relu": true},
    },
    // adapted from: https://github.com/soumith/kaggle_retinopathy_starter.torch/blob/master/models/googlenet_cudnn.lua
    "GoogLeNet2BN": []Conf{
        Conf{"warmup": 1},
        Conf{"layer": "DataLayer", "N": 128, "C": 3, "H": 224, "W": 224},
        Conf{"layer": "ConvLayer", "common": conv7, "K": 64, "bsum": true},
        Conf{"layer": "BatchNorm", "relu": true, "bsum": true},
        Conf{"layer": "PoolLayer", "common": pool3s2p1, "op": "max"},
        Conf{"layer": "ConvLayer", "common": conv1, "K": 64, "bsum": true},
        Conf{"layer": "BatchNorm", "relu": true, "bsum": true},
        Conf{"layer": "ConvLayer", "common": conv3, "K": 192, "bsum": true},
        Conf{"layer": "BatchNorm", "relu": true, "bsum": true},
        Conf{"layer": "PoolLayer", "common": pool3s2p1, "op": "max"},
        Inception2BN(64, 64, 64, 64, 96, "avg", 32),
        Inception2BN(64, 64, 96, 64, 96, "avg", 64),
        Inception2BN(0, 128, 160, 64, 96, "max", 0),
        Conf{"layer": "ConvLayer", "common": conv2, "K": 576, "bsum": true},
        Conf{"layer": "BatchNorm", "relu": true, "bsum": true},
        Inception2BN(224, 64, 96, 96, 128, "avg", 128),
        Inception2BN(192, 96, 128, 96, 128, "avg", 128),
        Inception2BN(160, 128, 160, 128, 160, "avg", 96),
        Inception2BN(96, 128, 192, 160, 192, "avg", 96),
        Inception2BN(0, 128, 192, 192, 256, "max",  0),
        Conf{"layer": "ConvLayer", "common": conv2, "K": 1024, "bsum": true},
        Conf{"layer": "BatchNorm", "relu": true, "bsum": true},
        Inception2BN(352, 192, 320, 160, 224, "avg", 128),
        Inception2BN(352, 192, 320, 192, 224, "max", 128),
        Conf{"layer": "PoolLayer", "common": pool7s1p0, "op": "avg"},
        Conf{"layer": "FullLayer", "nOut": 1000, "relu": true},
    },
}

//
//    Network runner
//

func RunNetwork(ng *cuda.CudaGenerator, net string, dtype base.Dtype) {
    network := networks[net]
    warmup := network[0].GetInt("warmup")
    network = network[1:]
    name := fmt.Sprintf("%s (dtype=%s, N=%d)", net, dtype, network[0].GetInt("N"))

    ng.WriteLine(`printf("------------------------------------------------\n");`)
    ng.WriteLine(`printf("Benchmarking: %s\n");`, name)
    ng.WriteLine(`printf("------------------------------------------------\n");`)

    var layers []cuda.Layer
    var prevLayer cuda.Layer
    prevLayerType := ""
    maxDeltas := 0
    maxWeights := 0
    var maxDeltaLayer cuda.Layer
    var maxWeightLayer cuda.Layer
    var sharedDeltas []backends.Tensor
    inception := false

    for _, conf := range network {
        layer := CreateLayer(ng, conf, prevLayer, dtype)

        layerType := conf.GetString("layer")
        if layerType == "Inception" {
            inception = true
        }

        // find the size of the largest buffers so they can be shared
        sizeF := layer.SizeF()
        if sizeF > maxWeights {
            maxWeights = sizeF
            maxWeightLayer = layer
        }

        sizeI := layer.SizeI()
        if sizeI > maxDeltas && prevLayerType != "DataLayer" {
            maxDeltas = sizeI
            maxDeltaLayer = layer
        }

        prevLayer = layer
        prevLayerType = layerType
        layers = append(layers, layer)
    }

    // Init shared buffers (assumes consistent dtype for now)
    deltaShape := maxDeltaLayer.DimI()
    deltaDtype := maxDeltaLayer.Dtype()
    sharedDeltas = append(sharedDeltas, ng.NewTensor(deltaShape, deltaDtype))
    sharedDeltas = append(sharedDeltas, ng.NewTensor(deltaShape, deltaDtype))
    if inception {
        sharedDeltas = append(sharedDeltas, ng.NewTensor(deltaShape, deltaDtype))
        sharedDeltas = append(sharedDeltas, ng.NewTensor(deltaShape, deltaDtype))
    }

    sharedUpdates := ng.NewTensor(maxWeightLayer.DimF(), base.Float32)

    for i, layer := range layers {
        if verbose {
            for _, line := range strings.Split(layer.String(), "\n") {
                ng.WriteLine(`printf("%s\n");`, line)
            }
        }

        // Intitalize buffers. Alernate shared delta buffer.
        // One layer can't have the same buffer for both error in and error out.
        layer.InitActivations(nil)
        layer.InitWeights(0.0, 0.1, sharedUpdates, zeros)
        if i > 1 {
            layer.InitDeltas(sharedDeltas)
        }
    }

    if verbose {
        ng.WriteLine("size_t remain, total;")
        ng.WriteLine("CudaMemGetInfo(remain, total);")
        ng.WriteLine("double gb = 1024.0 * 1024.0 * 1024.0;")
        ng.WriteLine(`printf("%%.3fGB of %%.3fGB Allocated (%%.3fGB Remaining)\n", `+
            `(total - remain) / gb, total / gb, remain / gb);`)
    }

    if zeros {
        layers[0].InitDataZero()
    } else {
        // give the first layer some data
        layers[0].InitDataUniform(0.0, 1.0)
/* DISABLED
        // Scale the initial weights so activations are bound around 1.0
        // We do this by running it through the forward pass and collecting mean stats
        ng.SetBench(False)
        var propagation backends.Tensor
        for _, layer := range layers {
            propagation = layer.Fprop(propagation, 0.5)
        }
        ng.SetBench(layerBench)
*/
    }

    ng.WriteLine("CudaEvent start;")
    ng.WriteLine("CudaEvent end;")

    ng.WriteLine("double fpropTime = 0.0;")
    ng.WriteLine("double bpropTime = 0.0;")
    ng.WriteLine("double fpropFlops = 0.0;")
    ng.WriteLine("double bpropFlops = 0.0;")

    ng.WriteLine("int loops = %d;", loops)

    // We throw away the first two runs as it includes cuda kernel loading times and clock warmup.
    // So add 1 to our loop count.
    ng.WriteLine("for (int i = 0; i < loops + %d; i++) {", warmup)
    ng.Indent(1)

    ng.WriteLine("int loop = i - %d + 1;", warmup)
    ng.WriteLine("if (loop < 0) {")
    ng.Indent(1)
    ng.WriteLine("loop = 0;")
    ng.Indent(-1)
    ng.WriteLine("}")

    ng.WriteLine("// Fprop")

    ng.WriteLine("start.Record();")
    flops := 0.0

    // fprop
    var propagation backends.Tensor
    for _, layer := range layers {
        propagation = layer.Fprop(propagation, 0.0)
        flops += layer.Flops()
/* NYI
        if printStats {
            layer.FpropStats()
        }
*/
    }

    ng.WriteLine("double flops = %g;", flops)

    ng.WriteLine("end.Record();")
    ng.WriteLine("end.Synchronize();")
    ng.WriteLine("double msecs = end.TimeSince(start);")
    ng.WriteLine(`printf("fprop(%%2d): %%8.3f msecs %%8.3f gflops\n", `+
        `loop, msecs, flops / (msecs * 1000000.0));`)

    ng.WriteLine("if (loop > 0) {")
    ng.Indent(1)
    ng.WriteLine("fpropTime += msecs;")
    ng.WriteLine("fpropFlops += flops;")
    ng.Indent(-1) 
    ng.WriteLine("}")

    ng.WriteLine("// Bprop")

    ng.WriteLine("start.Record();")
    flops = 0.0

    // bprop

    for  i := len(layers) - 1; i >= 0; i-- {
        layer := layers[i]
        propagation = layer.Bprop(propagation, 0.0)
        flops += layer.Flops() * 2
/* NYI
        if printStats {
            layer.BpropStats()
        }
*/
    }

    ng.WriteLine("flops = %g;", flops)

    ng.WriteLine("end.Record();")
    ng.WriteLine("end.Synchronize();")
    ng.WriteLine("msecs = end.TimeSince(start);")
    ng.WriteLine(`printf("bprop(%%2d): %%8.3f msecs %%8.3f gflops\n", `+
        `loop, msecs, flops / (msecs * 1000000.0));`)

    ng.WriteLine("if (loop > 0) {")
    ng.Indent(1)
    ng.WriteLine("bpropTime += msecs;")
    ng.WriteLine("bpropFlops += flops;")
    ng.Indent(-1) 
    ng.WriteLine("}")

    ng.Indent(-1)
    ng.WriteLine("}")

    if loops > 0 {
        ng.WriteLine(`printf("------------------------------------------------\n");`)
        ng.WriteLine(`printf("%s Results:\n");`, name)
        ng.WriteLine(`printf("------------------------------------------------\n");`)

        ng.WriteLine(`printf("Avg(%%d) fprop: %%8.3f msecs %%.3f gflops\n", `+
            `loops, fpropTime / loops, fpropFlops / (fpropTime * 1000000.0));`)
        ng.WriteLine(`printf("Avg(%%d) bprop: %%8.3f msecs %%.3f gflops\n", `+
            `loops, bpropTime / loops, bpropFlops / (bpropTime * 1000000.0));`)

        ng.WriteLine("fpropTime += bpropTime;")
        ng.WriteLine("fpropFlops += bpropFlops;")

        ng.WriteLine(`printf("Avg(%%d) total: %%8.3f msecs %%.3f gflops\n\n", `+
            `loops, fpropTime / loops, fpropFlops / (fpropTime * 1000000.0));`)
    }
}

//
//    Generation framework
//

func NewCudaGenerator() *cuda.CudaGenerator {
    ng := cuda.NewCudaGenerator(
        base.IntNone,            // rngSeed
        base.DtypeNone,          // defaultDtype
        base.IntNone,            // stochasticRound
        base.IntNone,            // deviceId
        computeCapability,       // computeCapability
        false,                   // bench
        base.IntNone,            // scratchSize
        base.IntNone,            // histBins
        base.IntNone,            // histOffset
        backends.CompatModeNone) // compatMode

    backends.SetBe(ng)

    return ng
}

//
//    Benchmark generator
//

func Benchmark(net string, dtype base.Dtype, outDir string) {
    ng := NewCudaGenerator()

    // start main function
    ng.StartCode()
    ng.WriteLine("void Main() {")
    ng.Indent(1)

    ng.WriteLine("Prolog();")

    RunNetwork(ng, net, dtype)

    // end main function
    ng.Indent(-1)
    ng.WriteLine("}")
    ng.WriteLine("")
    ng.EndCode()

    // create output directory if doesn't exist
    err := os.MkdirAll(outDir, 0777)
    if err != nil {
        base.RuntimeError("%s\n", err.Error())
    }

    // output generated code
    err = ng.OutputCode(outDir)
    if err != nil {
        base.RuntimeError("%s\n", err.Error())
    }
}

//
//    Main function
//

func main() {
    dtype := base.Float32
    Benchmark("Alexnet", dtype, "alexnet")
    Benchmark("AlexnetBN", dtype, "alexnet_bn")
    Benchmark("Overfeat", dtype, "overfeat")
    Benchmark("OverfeatBN", dtype, "overfeat_bn")
    Benchmark("VGG", dtype, "vgg")
    Benchmark("VGG_BN", dtype, "vgg_bn")
    Benchmark("VGG_E", dtype, "vgg_e")
    Benchmark("GoogLeNet1", dtype, "googlenet1")
    Benchmark("GoogLeNet1BN", dtype, "googlenet1_bn")
    Benchmark("GoogLeNet2", dtype, "googlenet2")
    Benchmark("GoogLeNet2BN", dtype, "googlenet2_bn")
}

