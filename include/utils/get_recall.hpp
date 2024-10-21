#ifndef INCLUDE_GET_RECALL_HPP
#define INCLUDE_GET_RECALL_HPP

#include <vector>
#include <stdlib.h>
#include <omp.h>
#include <unordered_set>

namespace utils {

/// @brief get the recall of the results
/// @param knn_results 
/// @param ground_truth 
/// @param dimension_gt the dimension of the ground_truth vector (cause the `ground_truth` is save in ivec(vector) file?)
/// @return 
template<typename TI>
double GetRecall(size_t k, size_t dimension_gt, const std::vector<id_t> & ground_truth, const std::vector<std::vector<TI>> & knn_results) {
    static_assert(std::is_same<TI, int32_t>::value || std::is_same<TI, uint32_t>::value ||
                std::is_same<TI, int64_t>::value || std::is_same<TI, uint64_t>::value, "Type index must be int32/uint32 or int64/uint64");
    const size_t nq = knn_results.size();
    size_t ok = 0;
    #pragma omp parallel for reduction(+: ok)
    for (size_t q = 0; q < nq; q++) {
        // const size_t actual_k = knn_results[q].size();
        std::unordered_set<id_t> st(ground_truth.begin() + (q + 0) * dimension_gt, ground_truth.begin() + (q + 0) * dimension_gt + k);
        for (const auto & id : knn_results[q]) {
        if (st.count(id)) {
            ok ++;
        }
        }
    }
    return double(ok) / (nq * k);
}

double GetRecall(size_t k, size_t dimension_gt, const std::vector<id_t> & ground_truth, const std::vector<id_t> & knn_results, id_t qid) {
  size_t ok = 0;
  std::unordered_set<id_t> st(ground_truth.begin() + qid * dimension_gt, ground_truth.begin() + qid * dimension_gt + k);
  for (const auto & id : knn_results) {
    if (st.count(id)) {
      ok ++;
    }
  }
  return double(ok) / (k);
}

}

#endif