
./build_lib_cpp.sh runtime
./build_lib_cpp.sh runtime_cpu
./build_lib_cuda.sh runtime_cuda

./build_prog_cuda.sh cifar10
./build_prog_cuda.sh cifar10_allcnn
./build_prog_cuda.sh cifar10_conv

./build_prog_cuda.sh mnist_branch
./build_prog_cuda.sh mnist_loadprm
./build_prog_cuda.sh mnist_merge
./build_prog_cuda.sh mnist_mlp
./build_prog_cuda.sh mnist_saveprm

./build_prog_cuda.sh conv_kernel_test

./build_convnet_benchmark.sh alexnet
./build_convnet_benchmark.sh alexnet_bn
./build_convnet_benchmark.sh overfeat
./build_convnet_benchmark.sh overfeat_bn
./build_convnet_benchmark.sh vgg
./build_convnet_benchmark.sh vgg_bn
./build_convnet_benchmark.sh vgg_e
./build_convnet_benchmark.sh googlenet1
./build_convnet_benchmark.sh googlenet1_bn
./build_convnet_benchmark.sh googlenet2
./build_convnet_benchmark.sh googlenet2_bn

./build_prog_cuda.sh caltech256_alexnet
./build_prog_cuda.sh caltech256_allcnn

./build_prog_cuda.sh imagenet_alexnet
./build_prog_cuda.sh imagenet_allcnn


