/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <IndexFlatCodes.h>

#include <AuxIndexStructures.h>
#include <FaissAssert.h>
#include <extra_distances.h>

namespace faiss {

IndexFlatCodes::IndexFlatCodes(size_t code_size, idx_t d, MetricType metric)
        : Index(d, metric), code_size(code_size) {}

IndexFlatCodes::IndexFlatCodes() : code_size(0) {}

void IndexFlatCodes::add(idx_t n, const float* x) {
    FAISS_THROW_IF_NOT(is_trained);
    if (n == 0) {
        return;
    }
    codes.resize((ntotal + n) * code_size);
    sa_encode(n, x, codes.data() + (ntotal * code_size));
    ntotal += n;
}

void IndexFlatCodes::reset() {
    codes.clear();
    ntotal = 0;
}

size_t IndexFlatCodes::sa_code_size() const {
    return code_size;
}

void IndexFlatCodes::reconstruct_n(idx_t i0, idx_t ni, float* recons) const {
    FAISS_THROW_IF_NOT(ni == 0 || (i0 >= 0 && i0 + ni <= ntotal));
    sa_decode(ni, codes.data() + i0 * code_size, recons);
}

void IndexFlatCodes::reconstruct(idx_t key, float* recons) const {
    reconstruct_n(key, 1, recons);
}

void IndexFlatCodes::check_compatible_for_merge(const Index& otherIndex) const {
    // minimal sanity checks
    const IndexFlatCodes* other =
            dynamic_cast<const IndexFlatCodes*>(&otherIndex);
    FAISS_THROW_IF_NOT(other);
    FAISS_THROW_IF_NOT(other->d == d);
    FAISS_THROW_IF_NOT(other->code_size == code_size);
    FAISS_THROW_IF_NOT_MSG(
            typeid(*this) == typeid(*other),
            "can only merge indexes of the same type");
}

void IndexFlatCodes::merge_from(Index& otherIndex, idx_t add_id) {
    FAISS_THROW_IF_NOT_MSG(add_id == 0, "cannot set ids in FlatCodes index");
    check_compatible_for_merge(otherIndex);
    IndexFlatCodes* other = static_cast<IndexFlatCodes*>(&otherIndex);
    codes.resize((ntotal + other->ntotal) * code_size);
    memcpy(codes.data() + (ntotal * code_size),
           other->codes.data(),
           other->ntotal * code_size);
    ntotal += other->ntotal;
    other->reset();
}

void IndexFlatCodes::permute_entries(const idx_t* perm) {
    std::vector<uint8_t> new_codes(codes.size());

    for (idx_t i = 0; i < ntotal; i++) {
        memcpy(new_codes.data() + i * code_size,
               codes.data() + perm[i] * code_size,
               code_size);
    }
    std::swap(codes, new_codes);
}

namespace {

} // anonymous namespace

} // namespace faiss
