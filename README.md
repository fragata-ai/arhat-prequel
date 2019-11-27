# Arhat

Arhat is an experimental deep learning framework implemented in [Go](https://golang.org/). Unlike most mainstream frameworks that perform training and inference computations directly, Arhat translates neural network descriptions into standalone lean executable platform-specific code. This approach allows direct embedding of the generated code into user applications and does not require deployment of sophisticated machine learning software stacks on the target platforms.

Arhat supports swappable platform-specific code generators (backends). Currently two backends are available:

* CPU: a reference backend generating C++ code for use on the CPU;
* CUDA: a backend generating C++/CUDA code for use on the NVIDIA GPU.

Implementation of more backends for other platforms is planned.

Arhat implementation is based on [neon](https://github.com/NervanaSystems/neon), Intel® Nervana™ reference deep learning framework. We have ported sections of neon code from Python to Go and partly redesigned it implementing code generators
in place of the original backends.

We use Arhat internally as a research platform for our larger ongoing project aiming at construction of a machine learning framework optimized for use on embedded platforms and massively parallel supercomputers.

## Requirements

The following hardware and software components are required for using Arhat:

* OS: Windows or Linux (macOS should suit too but was not yet tested);
* NVIDIA GPU device (Kepler of newer);
* CUDA Toolkit (8.0 or higher);
* C++ compiler toolchain supporting CUDA;
* Go (1.11 or higher);
* Python (only for obtaining data sets for examples).

## Obtaining data sets

Examples included in this distribution require two data sets:

* [MNIST](http://yann.lecun.com/exdb/mnist/)
* [CIFAR10](https://www.cs.toronto.edu/~kriz/cifar.html)

These data sets can be obtained and converted to Arhat format using the supplied Python scripts:

* `get_mnist.py`
* `get_cifar10.py`

## Runtime libraries

Arhat provides few run-time libraries required to build executables from the generated code:

* `runtime`: runtime components common for all backends;
* `runtime_cpu`: runtime components required for the CPU backend;
* `runtime_cuda`: runtime components required for the CUDA backend.

These libraries must be built and linked with the respective generated code.

## Quick start

To start, download this distribution, obtain the data sets, and build the runtime libraries. Set your `GOPATH` environment variable to the path of the downloaded `go` directory.

Create an empty working directory, make it you current directory, create a subdirectory named `data` and copy the data sets there. Then build the `mnist_mlp` example implementing a simple multi-layer perceptron network for MNIST data set. Use the following commands (Linux is assumed, commands for Windows are similar).

For the CUDA backend:

```
mkdir -p bin
mkdir -p mnist_mlp_cuda
go build -o bin/mnist_mlp_cuda fragata/arhat/examples/mnist_mlp_cuda
bin/mnist_mlp_cuda -o mnist_mlp_cuda
```

For the CPU backend:

```
mkdir -p bin
mkdir -p mnist_mlp_cpu
go build -o bin/mnist_mlp_cpu fragata/arhat/examples/mnist_mlp_cpu
bin/mnist_mlp_cpu -b cpu -o mnist_mlp_cpu
```

The generated code will be placed in subdirectories `mnist_mlp_cuda` and `mnist_mlp_cpu` respectively. Separately for each directory, compile all the contained source files and link resulting object files with the runtime libraries. Run the resulting executables from your current directory to train and evaluate the neural network.

## License

We are releasing Arhat under an open source [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0) License.

