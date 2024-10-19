#pragma once

#include <cuda.h>
#include <cublas.h>

__device__ inline float L2(float a, float b) {
    return (a - b) * (a - b);
}

__device__ inline float IP(float a, float b) {
    return a * b;
}