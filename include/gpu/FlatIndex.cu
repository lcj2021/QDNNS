/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <DeviceUtils.h>
#include <Distance.cuh>
#include <FlatIndex.cuh>
#include <L2Norm.cuh>
#include <ConversionOperators.cuh>
#include <CopyUtils.cuh>
#include <Transpose.cuh>

namespace faiss {
namespace gpu {

FlatIndex::FlatIndex(
        GpuResources* res,
        int dim,
        bool useFloat16,
        MemorySpace space)
        : resources_(res),
          dim_(dim),
          useFloat16_(useFloat16),
          space_(space),
          num_(0),
          rawData32_(
                  res,
                  AllocInfo(
                          AllocType::FlatData,
                          getCurrentDevice(),
                          space,
                          res->getDefaultStreamCurrentDevice())),
          rawData16_(
                  res,
                  AllocInfo(
                          AllocType::FlatData,
                          getCurrentDevice(),
                          space,
                          res->getDefaultStreamCurrentDevice())) {}

bool FlatIndex::getUseFloat16() const {
    return useFloat16_;
}

/// Returns the number of vectors we contain
idx_t FlatIndex::getSize() const {
    return vectors_.getSize(0);
}

int FlatIndex::getDim() const {
    return dim_;
}

void FlatIndex::reserve(size_t numVecs, cudaStream_t stream) {
    if (useFloat16_) {
    } else {
        rawData32_.reserve(numVecs * dim_ * sizeof(float), stream);
    }

    // The above may have caused a reallocation, we need to update the vector
    // types
    if (useFloat16_) {
    } else {
        DeviceTensor<float, 2, true> vectors32(
                (float*)rawData32_.data(), {num_, dim_});
        vectors_ = std::move(vectors32);
    }
}

Tensor<float, 2, true>& FlatIndex::getVectorsFloat32Ref() {
    // Should not call this unless we are in float32 mode
    FAISS_ASSERT(!useFloat16_);

    return vectors_;
}

void FlatIndex::query(
        Tensor<float, 2, true>& input,
        int k,
        faiss::MetricType metric,
        float metricArg,
        Tensor<float, 2, true>& outDistances,
        Tensor<idx_t, 2, true>& outIndices,
        bool exactDistance) {
    auto stream = resources_->getDefaultStreamCurrentDevice();

    if (useFloat16_) {
        // We need to convert the input to float16 for comparison to ourselves
    } else {
        bfKnnOnDevice(
                resources_,
                getCurrentDevice(),
                stream,
                vectors_,
                true, // is vectors row major?
                &norms_,
                input,
                true, // input is row major
                k,
                metric,
                metricArg,
                outDistances,
                outIndices,
                !exactDistance);
    }
}

void FlatIndex::reconstruct(
        idx_t start,
        idx_t num,
        Tensor<float, 2, true>& vecs) {
    auto stream = resources_->getDefaultStreamCurrentDevice();

    FAISS_ASSERT(vecs.getSize(0) == num);
    FAISS_ASSERT(vecs.getSize(1) == dim_);
}

void FlatIndex::reconstruct(
        Tensor<idx_t, 1, true>& ids,
        Tensor<float, 2, true>& vecs) {
    auto stream = resources_->getDefaultStreamCurrentDevice();

    FAISS_ASSERT(vecs.getSize(0) == ids.getSize(0));
    FAISS_ASSERT(vecs.getSize(1) == dim_);
}

void FlatIndex::add(const float* data, idx_t numVecs, cudaStream_t stream) {
    if (numVecs == 0) {
        return;
    }

    // convert and add to float16 data if needed
    if (useFloat16_) {
        // Make sure that `data` is on our device; we'll run the
        // conversion on our device
    } else {
        // add to float32 data
        rawData32_.append(
                (char*)data,
                (size_t)dim_ * numVecs * sizeof(float),
                stream,
                true /* reserve exactly */);
    }

    num_ += numVecs;

    if (useFloat16_) {
    } else {
        DeviceTensor<float, 2, true> vectors32(
                (float*)rawData32_.data(), {num_, dim_});
        vectors_ = std::move(vectors32);
    }

    // Precompute L2 norms of our database
    if (useFloat16_) {
    } else {
        DeviceTensor<float, 1, true> norms(
                resources_,
                makeSpaceAlloc(AllocType::FlatData, space_, stream),
                {num_});
        runL2Norm(vectors_, true, norms, true, stream);
        norms_ = std::move(norms);
    }
}

void FlatIndex::reset() {
    rawData32_.clear();
    rawData16_.clear();
    vectors_ = DeviceTensor<float, 2, true>();
    norms_ = DeviceTensor<float, 1, true>();
    num_ = 0;
}

} // namespace gpu
} // namespace faiss
