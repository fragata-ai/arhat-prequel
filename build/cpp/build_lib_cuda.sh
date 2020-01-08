#!/bin/bash

NAME=$1
LIB=lib

SRC=../../cpp/src
ARCH=sm_60

mkdir -p $LIB

nvcc -c -arch $ARCH -I $SRC $SRC/$NAME/*.cu

CC -c -O3 -I $SRC $SRC/$NAME/*.cpp
ar rcs $LIB/$NAME.a *.o

rm *.o

