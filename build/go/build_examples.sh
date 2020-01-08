
mkdir -p bin/examples

go build -o bin/examples/cifar10 fragata/arhat/examples/cifar10
go build -o bin/examples/cifar10_allcnn fragata/arhat/examples/cifar10_allcnn
go build -o bin/examples/cifar10_conv fragata/arhat/examples/cifar10_conv

go build -o bin/examples/mnist_branch fragata/arhat/examples/mnist_branch
go build -o bin/examples/mnist_loadprm fragata/arhat/examples/mnist_loadprm
go build -o bin/examples/mnist_merge fragata/arhat/examples/mnist_merge
go build -o bin/examples/mnist_mlp fragata/arhat/examples/mnist_mlp
go build -o bin/examples/mnist_saveprm fragata/arhat/examples/mnist_saveprm

go build -o bin/examples/conv_kernel_test fragata/arhat/examples/conv_kernel_test
go build -o bin/examples/convnet_benchmarks fragata/arhat/examples/convnet_benchmarks

go build -o bin/examples/caltech256_alexnet fragata/arhat/examples/caltech256_alexnet
go build -o bin/examples/caltech256_allcnn fragata/arhat/examples/caltech256_allcnn

go build -o bin/examples/imagenet_alexnet fragata/arhat/examples/imagenet_alexnet
go build -o bin/examples/imagenet_allcnn fragata/arhat/examples/imagenet_allcnn


