#pragma once

#include <cublas.h>

__device__ static inline float L2(float a, float b) {
    return (a - b) * (a - b);
}

__device__ static inline float IP(float a, float b) {
    return a * b;
}

__device__ static inline bool lt(float a, float b, int ia, int ib) {
    if (a != b) return a < b;
    return ia < ib;
}

__device__ static inline bool gt(float a, float b, int ia, int ib) {
    if (a != b) return a > b;
    return ia > ib;
}