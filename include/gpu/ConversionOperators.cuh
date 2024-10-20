/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <MetricType.h>
#include <DeviceTensor.cuh>

#include <cuda.h>
#include <thrust/execution_policy.h>
#include <thrust/transform.h>

namespace faiss {
namespace gpu {

//
// Conversion utilities
//

template <typename From, typename To>
struct Convert {
    inline __device__ To operator()(From v) const {
        return (To)v;
    }
};

template <typename T>
struct ConvertTo {};

template <>
struct ConvertTo<float> {
    static inline __device__ float to(float v) {
        return v;
    }
};

template <>
struct ConvertTo<float2> {
    static inline __device__ float2 to(float2 v) {
        return v;
    }
};

template <>
struct ConvertTo<float4> {
    static inline __device__ float4 to(float4 v) {
        return v;
    }
};

// Tensor conversion
template <typename From, typename To>
void runConvert(const From* in, To* out, size_t num, cudaStream_t stream) {
    thrust::transform(
            thrust::cuda::par.on(stream),
            in,
            in + num,
            out,
            Convert<From, To>());
}

template <typename From, typename To, int Dim>
void convertTensor(
        cudaStream_t stream,
        Tensor<From, Dim, true>& in,
        Tensor<To, Dim, true>& out) {
    FAISS_ASSERT(in.numElements() == out.numElements());

    runConvert<From, To>(in.data(), out.data(), in.numElements(), stream);
}

template <typename From, typename To, int Dim>
DeviceTensor<To, Dim, true> convertTensorTemporary(
        GpuResources* res,
        cudaStream_t stream,
        Tensor<From, Dim, true>& in) {
    FAISS_ASSERT(res);
    DeviceTensor<To, Dim, true> out(
            res, makeTempAlloc(AllocType::Other, stream), in.sizes());

    convertTensor(stream, in, out);
    return out;
}

template <typename From, typename To, int Dim>
DeviceTensor<To, Dim, true> convertTensorNonTemporary(
        GpuResources* res,
        cudaStream_t stream,
        Tensor<From, Dim, true>& in) {
    FAISS_ASSERT(res);
    DeviceTensor<To, Dim, true> out(
            res, makeDevAlloc(AllocType::Other, stream), in.sizes());

    convertTensor(stream, in, out);
    return out;
}

} // namespace gpu
} // namespace faiss
