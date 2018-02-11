#!/bin/bash

set -e

TF_LFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))') )
TF_CFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))') )

echo "Compiling CapsMatMul"
nvcc -std=c++11 -c -o capsmatmul.cu.o capsmatmul.cu.cc ${TF_CFLAGS[@]} \
    -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC --expt-relaxed-constexpr -ltensorflow_framework \
    -I /usr/local -I /usr/local/cuda/include -O3

echo "Compiling CapsMatMulGrad"
nvcc -std=c++11 -c -o capsmatmul_grad.cu.o capsmatmul_grad.cu.cc ${TF_CFLAGS[@]} \
    -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC --expt-relaxed-constexpr -ltensorfow_framework \
    -I /usr/local -I /usr/local/cuda/include -O3

g++ -std=c++11 -shared -o capsmatmul_op.so capsmatmul.cc capsmatmul.cu.o \
    capsmatmul_grad.cc capsmatmul_grad.cu.o ${TF_CFLAGS[@]} -fPIC -O3 \
    -L/usr/local/cuda/lib64 -lcudart ${TF_LFLAGS[@]}
