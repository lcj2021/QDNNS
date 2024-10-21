/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <IndexFlatCodes.h>

#include <AuxIndexStructures.h>
#include <DistanceComputer.h>
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

template <class VD>
struct GenericFlatCodesDistanceComputer : FlatCodesDistanceComputer {
    const IndexFlatCodes& codec;
    const VD vd;
    // temp buffers
    std::vector<uint8_t> code_buffer;
    std::vector<float> vec_buffer;
    const float* query = nullptr;

    GenericFlatCodesDistanceComputer(const IndexFlatCodes* codec, const VD& vd)
            : FlatCodesDistanceComputer(codec->codes.data(), codec->code_size),
              codec(*codec),
              vd(vd),
              code_buffer(codec->code_size * 4),
              vec_buffer(codec->d * 4) {}

    void set_query(const float* x) override {
        query = x;
    }

    float operator()(idx_t i) override {
        codec.sa_decode(1, codes + i * code_size, vec_buffer.data());
        return vd(query, vec_buffer.data());
    }

    float distance_to_code(const uint8_t* code) override {
        codec.sa_decode(1, code, vec_buffer.data());
        return vd(query, vec_buffer.data());
    }

    float symmetric_dis(idx_t i, idx_t j) override {
        codec.sa_decode(1, codes + i * code_size, vec_buffer.data());
        codec.sa_decode(1, codes + j * code_size, vec_buffer.data() + vd.d);
        return vd(vec_buffer.data(), vec_buffer.data() + vd.d);
    }

    void distances_batch_4(
            const idx_t idx0,
            const idx_t idx1,
            const idx_t idx2,
            const idx_t idx3,
            float& dis0,
            float& dis1,
            float& dis2,
            float& dis3) override {
        uint8_t* cp = code_buffer.data();
        for (idx_t i : {idx0, idx1, idx2, idx3}) {
            memcpy(cp, codes + i * code_size, code_size);
            cp += code_size;
        }
        // potential benefit is if batch decoding is more efficient than 1 by 1
        // decoding
        codec.sa_decode(4, code_buffer.data(), vec_buffer.data());
        dis0 = vd(query, vec_buffer.data());
        dis1 = vd(query, vec_buffer.data() + vd.d);
        dis2 = vd(query, vec_buffer.data() + 2 * vd.d);
        dis3 = vd(query, vec_buffer.data() + 3 * vd.d);
    }
};

struct Run_get_distance_computer {
    using T = FlatCodesDistanceComputer*;

    template <class VD>
    FlatCodesDistanceComputer* f(const VD& vd, const IndexFlatCodes* codec) {
        return new GenericFlatCodesDistanceComputer<VD>(codec, vd);
    }
};

template <class BlockResultHandler>
struct Run_search_with_decompress {
    using T = void;

    template <class VectorDistance>
    void f(VectorDistance& vd,
           const IndexFlatCodes* index_ptr,
           const float* xq,
           BlockResultHandler& res) {
        // Note that there seems to be a clang (?) bug that "sometimes" passes
        // the const Index & parameters by value, so to be on the safe side,
        // it's better to use pointers.
        const IndexFlatCodes& index = *index_ptr;
        size_t ntotal = index.ntotal;
        using SingleResultHandler =
                typename BlockResultHandler::SingleResultHandler;
        using DC = GenericFlatCodesDistanceComputer<VectorDistance>;
#pragma omp parallel // if (res.nq > 100)
        {
            std::unique_ptr<DC> dc(new DC(&index, vd));
            SingleResultHandler resi(res);
#pragma omp for
            for (int64_t q = 0; q < res.nq; q++) {
                resi.begin(q);
                dc->set_query(xq + vd.d * q);
                for (size_t i = 0; i < ntotal; i++) {
                    if (res.is_in_selection(i)) {
                        float dis = (*dc)(i);
                        resi.add_result(dis, i);
                    }
                }
                resi.end();
            }
        }
    }
};

} // anonymous namespace

FlatCodesDistanceComputer* IndexFlatCodes::get_FlatCodesDistanceComputer()
        const {
    Run_get_distance_computer r;
    return dispatch_VectorDistance(d, metric_type, metric_arg, r, this);
}

} // namespace faiss
