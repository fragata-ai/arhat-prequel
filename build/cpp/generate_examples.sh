
# cifar19

mkdir -p src/cifar10
mkdir -p src/cifar10_allcnn
mkdir -p src/cifar10_conv

rm -f src/cifar10/*
rm -f src/cifar10_allcnn/*
rm -f src/cifar10_conv/*

../go/bin/examples/cifar10 -o src/cifar10
../go/bin/examples/cifar10_allcnn -o src/cifar10_allcnn
../go/bin/examples/cifar10_conv -o src/cifar10_conv

# mnist

mkdir -p src/mnist_branch
mkdir -p src/mnist_loadprm
mkdir -p src/mnist_merge
mkdir -p src/mnist_mlp
mkdir -p src/mnist_saveprm

rm -f src/mnist_branch/*
rm -f src/mnist_loadprm/*
rm -f src/mnist_merge/*
rm -f src/mnist_mlp/*
rm -f src/mnist_saveprm/*

../go/bin/examples/mnist_branch -o src/mnist_branch
../go/bin/examples/mnist_loadprm -o src/mnist_loadprm
../go/bin/examples/mnist_merge -o src/mnist_merge
../go/bin/examples/mnist_mlp -o src/mnist_mlp
../go/bin/examples/mnist_saveprm -o src/mnist_saveprm

# conv_kernel_test

# TODO: Make output path configurable in conv_kernel_test

mkdir -p src/conv_kernel_test
rm -f src/conv_kernel_test/*

../go/bin/examples/conv_kernel_test

mv output/* src/conv_kernel_test
rmdir output

# convnet_benchmarks

../go/bin/examples/convnet_benchmarks

mkdir -p src/convnet_benchmarks
rm -f -R src/convnet_benchmarks/*

mv alexnet src/convnet_benchmarks
mv alexnet_bn src/convnet_benchmarks
mv overfeat src/convnet_benchmarks
mv overfeat_bn src/convnet_benchmarks
mv vgg src/convnet_benchmarks
mv vgg_bn src/convnet_benchmarks
mv vgg_e src/convnet_benchmarks
mv googlenet1 src/convnet_benchmarks
mv googlenet1_bn src/convnet_benchmarks
mv googlenet2 src/convnet_benchmarks
mv googlenet2_bn src/convnet_benchmarks

# caltech256

mkdir -p src/caltech256_alexnet
mkdir -p src/caltech256_allcnn

rm -f src/caltech256_alexnet/*
rm -f src/caltech256_allcnn/*

../go/bin/examples/caltech256_alexnet -o src/caltech256_alexnet -z 256 -e 90
../go/bin/examples/caltech256_allcnn -o src/caltech256_allcnn -z 256 -e 90

# imagenet

mkdir -p src/imagenet_alexnet
mkdir -p src/imagenet_allcnn

rm -f src/imagenet_alexnet/*
rm -f src/imagenet_allcnn/*

../go/bin/examples/imagenet_alexnet -o src/imagenet_alexnet -z 256 -e 90
../go/bin/examples/imagenet_allcnn -o src/imagenet_allcnn -z 256 -e 90


