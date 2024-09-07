#ifndef __SIMPLE_INCLUDE_H_H
#define __SIMPLE_INCLUDE_H_H

#include "cnrt.h"
#include <chrono>
//void MatMul_mlu_kernel(T *a,T *b,T *output,T m,T k,T n);

// template <typename T>
// void MatMul_mlu_kernel(T* a, T* b, T* output, T M,T K,T N);

// template <>
// void MatMul_mlu_kernel(size_t* a, size_t* b, size_t* output, size_t M,size_t K,size_t N);

// xx.h
template <typename T>
void MatMul_mlu_kernel(T* a, T* b, T* output, T M, T K, T N);

void MatMul_mlu_kernelV2(int8_t* a, int8_t* b, int8_t*  output, uint32_t M, uint32_t K, uint32_t N);

void testOp(float* input,int matmul_dim0, int matmul_dim1, int h, int w, int source_shape0, int kernel_shape0);
#endif