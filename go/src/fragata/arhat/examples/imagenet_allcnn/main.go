
package main

import (
    "fragata/arhat/backends"
    "fragata/arhat/generators/callbacks"
    "fragata/arhat/generators/data"
    "fragata/arhat/generators/models"
    "fragata/arhat/generators/optimizers"
    "fragata/arhat/generators/util"
    "fragata/arhat/initializers"
    "fragata/arhat/layers"
    "fragata/arhat/transforms"
)

var (
    Float = backends.NewFloat
    Callbacks = callbacks.NewCallbacks
    CannedImageData = data.NewCannedImageData
    Kaiming = initializers.NewKaiming
    Activation = layers.NewActivation
    Conv = layers.NewConv
    DataTransform = layers.NewDataTransform
    Dropout = layers.NewDropout
    Pooling = layers.NewPooling
    GeneralizedCost = layers.NewGeneralizedCost
    Model = models.NewModel
    GradientDescentMomentum = optimizers.NewGradientDescentMomentum
    DefaultSchedule = optimizers.NewDefaultSchedule
    Normalizer = transforms.NewNormalizer
    Rectlin = transforms.NewRectlin
    Softmax = transforms.NewSoftmax
    CrossEntropyMulti = transforms.NewCrossEntropyMulti
    TopKMisclassification = transforms.NewTopKMisclassification
    Engine = util.NewEngine
    ArgParser = util.NewArgParser
)

func main() {
    // parse the command line arguments
    parser := ArgParser()
    parser.Parse()
    args := parser.Args()

    engine := Engine(args...)
    engine.Run(Main)
}

func Main(engine *util.Engine) {
    dataset := CannedImageData(
        "path", engine.DataDir(), 
        "shape", []int{3, 224, 224},
        "nclass", 1000,
        "pixel_mean", []int{104, 112, 127},
        "source_range", []float64{0.0, 128.0},
        "target_range", []float64{0.0, 1.0})
    train := dataset.TrainIter()
    valid := dataset.ValidIter()

    mlayers := []layers.LayerItem{
/* Temporarily disabled: CUDA Convolution Bprop does not support C=3
        DataTransform("transform", Normalizer("divisor", 128.0)),
*/

        Conv(
            "fshape", []int{11, 11, 96}, 
            "init", Kaiming(), 
            "activation", Rectlin(), 
            "strides", 4, 
            "padding", 1),
        Conv(
            "fshape", []int{1, 1, 96}, 
            "init", Kaiming(), 
            "activation", Rectlin(), 
            "strides", 1),
        Conv(
            "fshape", []int{3, 3, 96}, 
            "init", Kaiming(), 
            "activation", Rectlin(), 
            "strides", 2, 
            "padding", 1), // 54->2,

        Conv(
            "fshape", []int{5, 5, 256}, 
            "init", Kaiming(), 
            "activation", Rectlin(), 
            "strides", 1), // 27->2,
        Conv(
            "fshape", []int{1, 1, 256}, 
            "init", Kaiming(), 
            "activation", Rectlin(), 
            "strides", 1),
        Conv(
            "fshape", []int{3, 3, 256}, 
            "init", Kaiming(), 
            "activation", Rectlin(), 
            "strides", 2, 
            "padding", 1), // 23->1,

        Conv(
            "fshape", []int{3, 3, 384}, 
            "init", Kaiming(), 
            "activation", Rectlin(), 
            "strides", 1, 
            "padding", 1),
        Conv(
            "fshape", []int{1, 1, 384}, 
            "init", Kaiming(), 
            "activation", Rectlin(), 
            "strides", 1),
        Conv(
            "fshape", []int{3, 3, 384}, 
            "init", Kaiming(), 
            "activation", Rectlin(), 
            "strides", 2, 
            "padding", 1), // 12->,

        Dropout("keep", 0.5),
        Conv(
            "fshape", []int{3, 3, 1024}, 
            "init", Kaiming(), 
            "activation", Rectlin(), 
            "strides", 1, 
            "padding", 1),
        Conv(
            "fshape", []int{1, 1, 1024}, 
            "init", Kaiming(), 
            "activation", Rectlin(), 
            "strides", 1),
        Conv(
            "fshape", []int{1, 1, 1000}, 
            "init", Kaiming(), 
            "activation", Rectlin(), 
            "strides", 1),
        Pooling("fshape", 6, "op", "avg"),
        Activation("transform", Softmax()),
    }

    model := Model(mlayers)

    schedWeight := DefaultSchedule("step_config", []int{10}, "change", 0.1)
    opt := GradientDescentMomentum(
        "learning_rate", 0.01, 
        "momentum_coef", 0.9, 
        "wdecay", 0.0005, 
        "schedule", schedWeight)

    // configure callbacks
    callbks := Callbacks()

    numEpochs := engine.Epochs()
    cost := GeneralizedCost("costfunc", CrossEntropyMulti())
    model.Fit(train, cost, opt, numEpochs, callbks)

    valmetric := TopKMisclassification(5)
    mets := model.Eval(valid, valmetric)

    engine.Display("Validation set metrics:")
    engine.Display("Logloss: %.2f, Accuracy: %.1f %% (Top-1), %.1f %% (Top-5)", 
        mets[0], 
        Float(1.0).Sub(mets[1]).Mul(Float(100.0)), 
        Float(1.0).Sub(mets[2]).Mul(Float(100.0)))
}

