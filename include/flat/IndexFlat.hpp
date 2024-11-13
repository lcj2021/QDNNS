#pragma once

#include <distance.hpp>
#include <iostream>
#include <utils/recall.hpp>
#include <algorithm>
#include <omp.h>

#ifdef _MSC_VER
#include <immintrin.h>
#else
#include <x86intrin.h>
#endif

namespace anns
{
namespace flat
{

    template <typename data_t>
    class IndexFlat
    {

    public:
        size_t D_;
        size_t num_threads_{1};
        
        std::vector<data_t> *base_vectors = nullptr;
        float (*distance)(const data_t *, const data_t *, size_t) = nullptr;

        IndexFlat(const std::vector<data_t>& base, size_t D_, float (*distance)(const data_t *, const data_t *, size_t)) 
            noexcept: D_(D_), distance(distance)
        {
            base_vectors = (std::vector<data_t>*)&base;    // std::vector<data_t>& base
        }

        void Search(const std::vector<std::vector<data_t>> &queries, size_t k, std::vector<std::vector<id_t>> &vids, std::vector<std::vector<float>> &dists)
        {
            size_t nq = queries.size();
            size_t nb = base_vectors->size() / D_;
            vids.clear();
            dists.clear();
            vids.resize(nq, std::vector<id_t>(k));
            dists.resize(nq, std::vector<data_t>(k));

#pragma omp parallel for schedule(dynamic, 1) num_threads(num_threads_)
            for (size_t qid = 0; qid < nq; ++qid)
            {
                const auto &query = queries[qid];
                auto &vid = vids[qid];
                auto &dist = dists[qid];

                std::vector<std::pair<data_t, id_t>> dists_candidates(nb);

                for (size_t i = 0; i < nb; ++i) {
                    dists_candidates[i].first = distance(query.data(), base_vectors->data() + i * D_, D_);
                    dists_candidates[i].second = i;
                }

                partial_sort(dists_candidates.begin(), dists_candidates.begin() + k, dists_candidates.end(), [](auto &l, auto &r) {
                    if (l.first != r.first) return l.first < r.first;
                    return l.second < r.second;
                });
                
                for (int i = 0; i < k; ++i) {
                    vids[qid][i] = dists_candidates[i].second;
                    dists[qid][i] = dists_candidates[i].first;
                }

                if (qid % 100 < 1) {
                    std::cerr << "Search " << qid << " / " << nq << std::endl;
                }
            }
        }

        size_t GetNumThreads() const noexcept
        {
            return num_threads_;
        }

        void SetNumThreads(size_t num_threads) noexcept
        {
            num_threads_ = num_threads;
        }
    };

}; // namespace flat
}; // namespace index
