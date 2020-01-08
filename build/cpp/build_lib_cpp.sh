#!/bin/bash

NAME=$1
LIB=lib

SRC=../../cpp/src

mkdir -p $LIB

CC -c -O3 -I $SRC $SRC/$NAME/*.cpp
ar rcs $LIB/$NAME.a *.o

rm *.o

