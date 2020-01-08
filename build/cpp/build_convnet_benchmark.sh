
NAME=$1
SRC=src/convnet_benchmarks
LIB=lib
BIN=bin

SRCL=../../cpp/src
ARCH=sm_60

mkdir -p $BIN

nvcc -c -arch $ARCH -I $SRC -I $SRCL $SRC/$NAME/*.cu
CC -c -O3 -I $SRC -I $SRCL $SRC/$NAME/*.cpp

CC -o $BIN/$NAME *.o \
    $LIB/runtime_cuda.a $LIB/runtime.a \
    -lcublas -lcurand -lcudart

rm *.o

