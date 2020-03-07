#!/bin/bash
set -e
set -x
TENSORRT_DIR=../../TensorRT-7.0.0.11
g++ -Wall -Wno-deprecated-declarations -std=c++11  -I"../common" -I"/usr/local/cuda/include" -I"/usr/local/cuda/include" -I"${TENSORRT_DIR}/include" -D_REENTRANT -c -o mnist.o mnist.cpp
g++ -Wall -Wno-deprecated-declarations -std=c++11  -I"../common" -I"/usr/local/cuda/include" -I"/usr/local/cuda/include" -I"${TENSORRT_DIR}/include" -D_REENTRANT -c -o logger.o logger.cpp
g++ -o ./mnist ./mnist.o ./logger.o -L"/usr/local/cuda/lib64" -L"/usr/local/cuda/lib64" -L"${TENSORRT_DIR}/lib" -Wl,--start-group -lnvinfer -lnvparsers -lnvinfer_plugin -lnvonnxparser -lcudnn -lcublas -lcudart -lrt -ldl -lpthread -lmyelin -lnvrtc -Wl,--end-group
