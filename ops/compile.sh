#!/bin/bash

set -e

TF_LFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))') )
TF_CFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))') )

echo "Compiling CapsulePrediction"
nvcc -std=c++11 -c -o capsuleprediction.cu.o capsuleprediction.cu.cc ${TF_CFLAGS[@]} \
    -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC --expt-relaxed-constexpr -ltensorflow_framework \
    -I /usr/local -I /usr/local/cuda/include -O3

echo "Compiling CapsulePredictionGrad"
nvcc -std=c++11 -c -o capsuleprediction_grad.cu.o capsuleprediction_grad.cu.cc ${TF_CFLAGS[@]} \
    -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC --expt-relaxed-constexpr -ltensorfow_framework \
    -I /usr/local -I /usr/local/cuda/include -O3

g++ -std=c++11 -shared -o capsuleprediction_op.so capsuleprediction.cc capsuleprediction.cu.o \
    capsuleprediction_grad.cc capsuleprediction_grad.cu.o ${TF_CFLAGS[@]} -fPIC -O3 \
    -L/usr/local/cuda/lib64 -lcudart ${TF_LFLAGS[@]}

echo "Done!"