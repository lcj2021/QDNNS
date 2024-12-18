#pragma once

#include <distance.hpp>
#include <random>
#include <iostream>
#include <fstream>
#include <string>
#include <queue>
#include <memory>
#include <map>
#include <mutex>
#include <utils/binary_io.hpp>
#include <utils/resize.hpp>
#include <utils/recall.hpp>
#include <algorithm>
#include <stdexcept>
#include <limits>
#include <LightGBM/c_api.h>

#ifdef _MSC_VER
#include <immintrin.h>
#else
#include <x86intrin.h>
#endif

#include <atomic>
#include <omp.h>
#define MAGIC_VEC_ID std::numeric_limits<id_t>::max()

std::map<std::string, int> dataset_threshold {
    {"deep100m", 979},
    {"wikipedia", 951},
    {"datacomp-image", 699},
    {"spacev100m", 800},
};

namespace anns
{

  namespace graph
  {
    struct IntermediateResult
    {
        // std::priority_queue<std::pair<float, id_t>> top_candidates, candidate_set;
        bool ready = false;
        // std::vector<id_t> visited;
        // float low_bound;
        // size_t NDC = 0;
        id_t enterpoint_node = MAGIC_VEC_ID;
        IntermediateResult(size_t check_stamp)
        {
            // top_candidates = std::priority_queue<std::pair<float, id_t>>();
            // candidate_set = std::priority_queue<std::pair<float, id_t>>();
            ready = false;
            // visited.reserve(check_stamp);
            // low_bound = std::numeric_limits<float>::max();
            // NDC = 0;
            enterpoint_node = MAGIC_VEC_ID;
        }
    };

    template <typename data_t>
    class HNSW
    {

    public:

      size_t cur_element_count_{0};
      size_t D_{0}; // vector dimensions
      size_t Mmax_{0};            // maximum number of connections for each element per layer
      size_t Mmax0_{0};           // maximum number of connections for each element in layer0
      size_t ef_construction_{0}; // usually been set to 128
      double mult_{0.0};
      double rev_size_{0.0};
      int max_level_{0};
      id_t enterpoint_node_{MAGIC_VEC_ID};
      int random_seed_{100};
      float (*distance)(const data_t *, const data_t *, size_t) = nullptr;

      std::vector<int> element_levels_; // keeps level of each element
      std::vector<std::vector<std::vector<id_t>>> link_lists_;
      std::default_random_engine level_generator_;
      std::vector<std::unique_ptr<std::mutex>> link_list_locks_;
      std::vector<const data_t*> data_memory_; // vector data start pointer of memory.

      std::mutex global_;
      size_t num_threads_{1};
      std::atomic<size_t> comparison_{0};
      
      std::string dataset, prefix = "/home/zhengweiguo/liuchengjun/";

      std::vector<std::vector<float>> train_feats_nn;
      std::vector<std::vector<float>> test_feats_nn;
      std::vector<std::vector<float>> train_feats_lgb;
      std::vector<std::vector<float>> test_feats_lgb;
      std::vector<std::vector<int>> train_label;
      std::vector<std::vector<int>> test_label;
      std::vector<IntermediateResult> test_inter_results;

      std::vector<id_t> test_gt, train_gt;
      id_t dimension_gt, num_test, num_train;

      size_t check_stamp = 1000;

        BoosterHandle handle; // LightGBM model
        bool lgb_ready = false;
      size_t num_check = 100;     // 
      int recall_at_k = 100;

      HNSW(size_t D, size_t max_elements, size_t M, size_t ef_construction, 
      std::string dataset,
      int recall_at_k,
      size_t check_stamp,
      float (*distance)(const data_t *, const data_t *, size_t),
      size_t random_seed = 100) noexcept: 
        D_(D), Mmax_(M), Mmax0_(2 * M), ef_construction_(std::max(ef_construction, M)), random_seed_(random_seed), mult_(1 / log(1.0 * Mmax_)), rev_size_(1.0 / mult_), recall_at_k(recall_at_k), check_stamp(check_stamp), distance(distance)
      {
        level_generator_.seed(random_seed);

        // std::string prefix = dataset + "."
        //     "recall_at_" + std::to_string(recall_at_k) + "." +
        //     "ck_ts_" + std::to_string(check_stamp) + "." + 
        //     "ncheck_" + std::to_string(num_check);

        // std::cout << "check_stamp: " << check_stamp << std::endl;

        // std::tie(num_test, dimension_gt) = utils::LoadFromFile(test_gt, "/home/zhengweiguo/liuchengjun/anns/query/" + dataset + "/base_gt.ivecs");
        // std::tie(num_train, dimension_gt) = utils::LoadFromFile(train_gt, "/home/zhengweiguo/liuchengjun/anns/dataset/" + dataset + "/learn_gt.ivecs");
      }

        HNSW(const std::vector<data_t>& base, const std::string& filename, std::string dataset,
        int recall_at_k,
        size_t check_stamp,
        float (*distance)(const data_t *, const data_t *, size_t),
        size_t random_seed = 100) noexcept: distance(distance)
        {
            std::cout << "[HNSW] Loading HNSW from file: " << filename << std::endl;
            std::ifstream in(filename, std::ios::binary);
            in.read(reinterpret_cast<char*>(&cur_element_count_), sizeof(cur_element_count_));
            in.read(reinterpret_cast<char*>(&D_), sizeof(D_));
            in.read(reinterpret_cast<char*>(&Mmax_), sizeof(Mmax_));
            in.read(reinterpret_cast<char*>(&Mmax0_), sizeof(Mmax0_));
            in.read(reinterpret_cast<char*>(&ef_construction_), sizeof(ef_construction_));
            in.read(reinterpret_cast<char*>(&mult_), sizeof(mult_));
            in.read(reinterpret_cast<char*>(&rev_size_), sizeof(rev_size_));
            in.read(reinterpret_cast<char*>(&max_level_), sizeof(max_level_));
            in.read(reinterpret_cast<char*>(&enterpoint_node_), sizeof(enterpoint_node_));
            in.read(reinterpret_cast<char*>(&random_seed_), sizeof(random_seed_));
            element_levels_.resize(cur_element_count_);
            in.read(reinterpret_cast<char*>(element_levels_.data()), cur_element_count_ * sizeof(int));
            link_lists_.resize(cur_element_count_);
            for (id_t id = 0; id < cur_element_count_; id++)
            {
                auto& ll = link_lists_[id];
                ll.resize(element_levels_[id] + 1);
                for (auto& l: ll)
                {
                    size_t n;
                    in.read(reinterpret_cast<char*>(&n), sizeof(size_t));
                    l.resize(n);
                    in.read(reinterpret_cast<char*>(l.data()), n * sizeof(id_t));
                }
            }
            level_generator_.seed(random_seed_);
            link_list_locks_.resize(cur_element_count_);
            std::for_each(link_list_locks_.begin(), link_list_locks_.end(), [](std::unique_ptr<std::mutex>& lock) 
            { lock = std::make_unique<std::mutex>(); });
            data_memory_.reserve(cur_element_count_);
            for (size_t i = 0; i < cur_element_count_; i++)
            {
                data_memory_.emplace_back(base.data() + i * D_);
            }

            this->recall_at_k = recall_at_k;
            this->check_stamp = check_stamp;
            this->dataset = dataset;
            num_test = 10000;
            
            std::string test_gt_path, train_gt_path, dataname = dataset.substr(0, dataset.size() - 5);
            if (dataset == "imagenet.base" || dataset == "wikipedia.base"
                || dataset == "datacomp-image.base" || dataset == "datacomp-text.base") {
                if (dataset == "datacomp-image.base") {
                    test_gt_path = prefix + "anns/query/" + "datacomp-text" + "/query.t2i.base.norm.gt.ivecs.cpu.1000";
                    train_gt_path = prefix + "anns/query/" + "datacomp-text" + "/learn.t2i.base.norm.gt.ivecs.cpu.1000";
                } else if (dataset == "datacomp-text.base") {
                    test_gt_path = prefix + "anns/query/" + "datacomp-image" + "/query.i2t.base.norm.gt.ivecs.cpu.1000";
                    train_gt_path = prefix + "anns/query/" + "datacomp-image" + "/learn.i2t.base.norm.gt.ivecs.cpu.1000";
                } else {
                    test_gt_path = "/home/zhengweiguo/liuchengjun/anns/query/" + dataname + "/query.norm.gt.ivecs.cpu.1000";   
                    train_gt_path = "/home/zhengweiguo/liuchengjun/anns/query/" + dataname + "/learn.norm.gt.ivecs.cpu.1000";
                }
            } else {
                test_gt_path = "/home/zhengweiguo/liuchengjun/anns/query/" + dataname + "/query.gt.ivecs.cpu.1000";   
                train_gt_path = "/home/zhengweiguo/liuchengjun/anns/query/" + dataname + "/learn.gt.ivecs.cpu.1000";
            }
            std::tie(num_test, dimension_gt) = utils::LoadFromFile(test_gt, test_gt_path);
            std::tie(num_train, dimension_gt) = utils::LoadFromFile(train_gt, train_gt_path);
            dimension_gt = 1000;
            num_test /= dimension_gt;
            num_train /= dimension_gt;
            std::cout << "num_test: " << num_test << std::endl;
            std::cout << "num_train: " << num_train << std::endl;

            train_feats_nn.resize(num_train);
            train_feats_lgb.resize(num_train);
            train_label.resize(num_train);
            test_feats_nn.resize(num_test);
            test_feats_lgb.resize(num_test);
            test_label.resize(num_test);
            test_inter_results.resize(num_test, IntermediateResult(check_stamp));
        }

      void Save(const std::string& filename) const noexcept
      {
        std::ofstream out(filename, std::ios::binary);
        out.write(reinterpret_cast<const char*>(&cur_element_count_), sizeof(cur_element_count_));
        out.write(reinterpret_cast<const char*>(&D_), sizeof(D_));
        out.write(reinterpret_cast<const char*>(&Mmax_), sizeof(Mmax_));
        out.write(reinterpret_cast<const char*>(&Mmax0_), sizeof(Mmax0_));
        out.write(reinterpret_cast<const char*>(&ef_construction_), sizeof(ef_construction_));
        out.write(reinterpret_cast<const char*>(&mult_), sizeof(mult_));
        out.write(reinterpret_cast<const char*>(&rev_size_), sizeof(rev_size_));
        out.write(reinterpret_cast<const char*>(&max_level_), sizeof(max_level_));
        out.write(reinterpret_cast<const char*>(&enterpoint_node_), sizeof(enterpoint_node_));
        out.write(reinterpret_cast<const char*>(&random_seed_), sizeof(random_seed_));
        const char* buffer = reinterpret_cast<const char*>(element_levels_.data());
        out.write(buffer, element_levels_.size() * sizeof(int));
        for (const auto& ll: link_lists_)
        {
          for (const auto& l: ll)
          {
            size_t n = l.size();
            const char* buffer = reinterpret_cast<const char*>(l.data());
            out.write(reinterpret_cast<const char*>(&n), sizeof(n));
            out.write(buffer, sizeof(id_t) * n);
          }
        }
      }
      
      /// @brief  Add a point to the graph [User should not call this function directly]
      /// @param data_point 
      void BuildPoint(id_t cur_id, const data_t *data_point)
      {
        // Write the data point
        data_memory_[cur_id] = data_point;

        // alloc memory for the link lists
        std::unique_lock<std::mutex> lock_el(*link_list_locks_[cur_id]);
        int cur_level = GetRandomLevel(mult_);
        for (int lev = 0; lev <= cur_level; lev++)
        {
          link_lists_[cur_id].emplace_back(std::vector<id_t>());
        }
        element_levels_[cur_id] = cur_level;

        std::unique_lock<std::mutex> temp_lock(global_);
        int max_level_copy = max_level_;
        id_t cur_obj = enterpoint_node_;
        id_t enterpoint_node_copy = enterpoint_node_;
        if (cur_level <= max_level_) temp_lock.unlock();

        if (enterpoint_node_copy != MAGIC_VEC_ID)
        { // not first element
          if (cur_level < max_level_copy)
          {
            // find the closet node in upper layers
            float cur_dist = distance(data_point, data_memory_[cur_obj], D_);
            for (int lev = max_level_copy; lev > cur_level; lev--)
            {
              bool changed = true;
              while (changed)
              {
                changed = false;
                std::unique_lock<std::mutex> wlock(*link_list_locks_[cur_obj]);
                const auto& neighbors = link_lists_[cur_obj][lev];
                size_t num_neighbors = neighbors.size();

                for (size_t i = 0; i < num_neighbors; i++)
                {
                  id_t cand = neighbors[i];
                  float d = distance(data_point, data_memory_[cand], D_);
                  if (d < cur_dist)
                  {
                    cur_dist = d;
                    cur_obj = cand;
                    changed = true;
                  }
                }
              }
            }
          }
          /// add edges to lower layers from the closest node
          for (int lev = std::min(cur_level, max_level_copy); lev >= 0; lev--)
          {
            auto top_candidates = SearchBaseLayer(cur_obj, data_point, lev, ef_construction_);
            cur_obj = MutuallyConnectNewElement(data_point, cur_id, top_candidates, lev);
          }
        }
        else
        {
          // Do nothing for the first element
          enterpoint_node_ = cur_id;
          max_level_ = cur_level;
        }

        // Releasing lock for the maximum level
        if (cur_level > max_level_copy)
        {
          enterpoint_node_ = cur_id;
          max_level_ = cur_level;
        }
      }

      void BuildIndex(const std::vector<data_t> &raw_data)
      {
        size_t num_points = raw_data.size() / D_;
        cur_element_count_ = num_points;

        data_memory_.resize(num_points);
        link_lists_.resize(num_points);
        link_list_locks_.resize(num_points);
        std::for_each(link_list_locks_.begin(), link_list_locks_.end(), [](auto &lock){ lock = std::make_unique<std::mutex>(); });
        element_levels_.resize(num_points, 0);

#pragma omp parallel for schedule(dynamic, 1) num_threads(num_threads_)
        for (id_t id = 0; id < num_points; id++)
        {
            if (rand() % 100000 < 1) {
                std::cout << "Building " << id << " / " << num_points << std::endl;
            }
            BuildPoint(id, raw_data.data() + id * D_);
        }
      }

      void BuildIndex(const std::vector<const data_t *> &raw_data)
      {
        size_t num_points = raw_data.size();
        cur_element_count_ = num_points;

        data_memory_.resize(num_points);
        link_lists_.resize(num_points);
        link_list_locks_.resize(num_points);
        std::for_each(link_list_locks_.begin(), link_list_locks_.end(), [](auto &lock){ lock = std::make_unique<std::mutex>(); });
        element_levels_.resize(num_points, 0);

#pragma omp parallel for schedule(dynamic, 1) num_threads(num_threads_)
        for (id_t id = 0; id < num_points; id++)
        {
            if (rand() % 100000 < 1) {
                std::cout << "Building " << id << " / " << num_points << std::endl;
            }
            BuildPoint(id, raw_data[id]);
        }
      }

      std::priority_queue<std::pair<float, id_t>> Search(const data_t *query_data, size_t k, size_t ef)
      {
        assert(ef >= k && "ef > k!");

        if (cur_element_count_ == 0)
          return std::priority_queue<std::pair<float, id_t>>();

        size_t comparison = 0;
        id_t cur_obj = enterpoint_node_;
        float cur_dist = distance(query_data, data_memory_[enterpoint_node_], D_);
        comparison++;

        for (int lev = element_levels_[enterpoint_node_]; lev > 0; lev--)
        {
          // find the closet node in upper layers
          bool changed = true;
          while (changed)
          {
            changed = false;
            const auto& neighbors = link_lists_[cur_obj][lev];
            size_t num_neighbors = neighbors.size();

            for (size_t i = 0; i < num_neighbors; i++)
            {
              id_t cand = neighbors[i];
              float d = distance(query_data, data_memory_[cand], D_);
              if (d < cur_dist)
              {
                cur_dist = d;
                cur_obj = cand;
                changed = true;
              }
            }
            comparison += num_neighbors;
          }
        }

        auto top_candidates = SearchBaseLayer(cur_obj, query_data, 0, ef);

        while (top_candidates.size() > k)
        {
          top_candidates.pop();
        }

        comparison_.fetch_add(comparison);

        return top_candidates;
      }

      std::priority_queue<std::pair<float, id_t>> Search(const data_t *query_data, size_t k, size_t ef, id_t ep)
      {
        assert(ef >= k && "ef > k!");

        if (cur_element_count_ == 0)
          return std::priority_queue<std::pair<float, id_t>>();

        size_t comparison = 0;

        id_t cur_obj = ep;
        float cur_dist = distance(query_data, data_memory_[enterpoint_node_], D_);
        comparison++;

        for (int lev = element_levels_[ep]; lev > 0; lev--)
        {
          // find the closet node in upper layers
          bool changed = true;
          while (changed)
          {
            changed = false;
            const auto& neighbors = link_lists_[cur_obj][lev];
            size_t num_neighbors = neighbors.size();

            for (size_t i = 0; i < num_neighbors; i++)
            {
              id_t cand = neighbors[i];
              float d = distance(query_data, data_memory_[cand], D_);
              if (d < cur_dist)
              {
                cur_dist = d;
                cur_obj = cand;
                changed = true;
              }
            }
            comparison += num_neighbors;
          }
        }

        auto top_candidates = SearchBaseLayer(cur_obj, query_data, 0, ef);

        while (top_candidates.size() > k)
        {
          top_candidates.pop();
        }

        comparison_.fetch_add(comparison);

        return top_candidates;
      }

        void Search(const std::vector<std::vector<data_t>> &queries, size_t k, size_t ef, std::vector<std::vector<id_t>> &vids, std::vector<std::vector<float>> &dists)
        {
            size_t nq = queries.size();
            vids.clear();
            dists.clear();
            vids.resize(nq);
            dists.resize(nq);

    #pragma omp parallel for schedule(dynamic, 1) num_threads(num_threads_)
            for (size_t i = 0; i < nq; i++)
            {
                const auto &query = queries[i];
                auto &vid = vids[i];
                auto &dist = dists[i];

                auto r = Search(query.data(), k, ef);
                vid.reserve(r.size());
                dist.reserve(r.size());
                while (r.size())
                {
                    const auto &te = r.top();
                    vid.emplace_back(te.second);
                    dist.emplace_back(te.first);
                    r.pop();
                }
                std::reverse(vid.begin(), vid.end());
                if (rand() % 10000 < 1) {
                    std::cerr << "Search " << i << " / " << nq << std::endl;
                }
            }
        }

        void Search(const std::vector<data_t> &queries, size_t k, size_t ef, std::vector<std::vector<id_t>> &vids, std::vector<std::vector<float>> &dists)
        {
            size_t nq = queries.size() / D_;
            vids.clear();
            dists.clear();
            vids.resize(nq);
            dists.resize(nq);

    #pragma omp parallel for schedule(dynamic, 1) num_threads(num_threads_)
            for (size_t i = 0; i < nq; i++)
            {
                auto &vid = vids[i];
                auto &dist = dists[i];

                auto r = Search(queries.data() + i * D_, k, ef);
                vid.reserve(r.size());
                dist.reserve(r.size());
                while (r.size())
                {
                    const auto &te = r.top();
                    vid.emplace_back(te.second);
                    dist.emplace_back(te.first);
                    r.pop();
                }
                std::reverse(vid.begin(), vid.end());
                if (rand() % 10000 < 1) {
                    std::cerr << "Search " << i << " / " << nq << std::endl;
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

      size_t GetComparisonAndClear() noexcept
      {
        return comparison_.exchange(0);
      }

      size_t IndexSize() const noexcept
      {
        size_t sz = 0;
        for (const auto & ll: link_lists_) {
          for (const auto & l: ll) {
            sz += l.size() * sizeof(id_t);
          }
        }
        return sz;
      }

      /// @brief Connection new element and return next cloest element id
      /// @param data_point
      /// @param id
      /// @param top_candidates
      /// @param layer
      /// @return
      id_t MutuallyConnectNewElement(const data_t *data_point, id_t id, std::priority_queue<std::pair<float, id_t>> &top_candidates, int level)
      {
        size_t Mcurmax = level ? Mmax_ : Mmax0_;
        PruneNeighbors(top_candidates, Mcurmax);

        auto& neighbors_cur = link_lists_[id][level];
        /// @brief Edge-slots check and Add neighbors for current vector
        {
          // lock only during the update
          // because during the addition the lock for cur_c is already acquired
          std::unique_lock<std::mutex> lock(*link_list_locks_[id], std::defer_lock);
          neighbors_cur.clear();
          assert(top_candidates.size() <= Mcurmax);
          neighbors_cur.reserve(top_candidates.size());
          
          while (top_candidates.size())
          {
            neighbors_cur.emplace_back(top_candidates.top().second);
            top_candidates.pop();
          }
        }

        id_t next_closet_entry_point = neighbors_cur.back();

        for (id_t sid: neighbors_cur)
        {
          std::unique_lock<std::mutex> lock(*link_list_locks_[sid]);

          auto& neighbors = link_lists_[sid][level];
          size_t sz_link_list_other = neighbors.size();

          if (sz_link_list_other > Mcurmax)
          {
            std::cout << sz_link_list_other << ">" << Mcurmax << std::endl;
            std::cerr << "Bad value of sz_link_list_other" << std::endl;
            exit(1);
          }
          if (sid == id)
          {
            std::cerr << "Trying to connect an element to itself" << std::endl;
            exit(1);
          }
          if (level > element_levels_[sid])
          {
            std::cerr << "Trying to make a link on a non-existent level" << std::endl;
            exit(1);
          }

          if (sz_link_list_other < Mcurmax)
          {
            neighbors.emplace_back(id);
          }
          else
          {
            // finding the "farest" element to replace it with the new one
            float d_max = distance(data_memory_[id], data_memory_[sid], D_);
            // Heuristic:
            std::priority_queue<std::pair<float, id_t>> candidates;
            candidates.emplace(d_max, id);

            for (size_t j = 0; j < sz_link_list_other; j++)
            {
              candidates.emplace(distance(data_memory_[neighbors[j]], data_memory_[sid], D_), neighbors[j]);
            }

            PruneNeighbors(candidates, Mcurmax);
            // Copy neighbors and add edges
            neighbors.clear();
            neighbors.reserve(candidates.size());
            while (candidates.size())
            {
              neighbors.emplace_back(candidates.top().second);
              candidates.pop();
            }
          }
        }

        return next_closet_entry_point;
      }

      /// @brief Return max heap of the top NN elements
      /// @param top_candidates 
      /// @param NN 
      void PruneNeighbors(std::priority_queue<std::pair<float, id_t>> &top_candidates, size_t NN)
      {
        if (top_candidates.size() < NN)
        {
          return;
        }

        std::priority_queue<std::pair<float, id_t>> queue_closest; // min heap
        std::vector<std::pair<float, id_t>> return_list;

        while (top_candidates.size())
        { // replace top_candidates with a min-heap, so that each poping can return the nearest neighbor.
          const auto &te = top_candidates.top();
          queue_closest.emplace(-te.first, te.second);
          top_candidates.pop();
        }

        while (queue_closest.size())
        {
          if (return_list.size() >= NN)
          {
            break;
          }

          const auto curen = queue_closest.top();
          float dist2query = -curen.first;
          const data_t *curenv = data_memory_[curen.second];
          queue_closest.pop();
          bool good = true;
          for (const auto &curen2 : return_list)
          {
            float dist2curenv2 = distance(data_memory_[curen2.second], curenv, D_);
            if (dist2curenv2 < dist2query)
            {
              good = false;
              break;
            }
          }
          if (good)
          {
            return_list.emplace_back(curen);
          }
        }

        for (const auto &elem : return_list)
        {
          top_candidates.emplace(-elem.first, elem.second);
        }
      }

      int GetRandomLevel(double reverse_size)
      {
        std::uniform_real_distribution<double> distribution(0.0, 1.0);
        double r = -log(distribution(level_generator_)) * reverse_size;
        return (int)r;
      }

      /// @brief Return the topk nearest neighbors (max-heap) of a given data point on a certain level
      /// @param ep_id 
      /// @param data_point 
      /// @param level 
      /// @param ef 
      /// @return 
      std::priority_queue<std::pair<float, id_t>> SearchBaseLayer(id_t ep_id, const data_t *data_point, int level, size_t ef)
      {
        size_t comparison = 0;
        std::vector<bool> mass_visited(cur_element_count_, false);
        std::priority_queue<std::pair<float, id_t>> top_candidates;
        std::priority_queue<std::pair<float, id_t>> candidate_set;
        float dist = distance(data_point, data_memory_[ep_id], D_);
        comparison++;
        top_candidates.emplace(dist, ep_id); // max heap
        candidate_set.emplace(-dist, ep_id); // min heap
        mass_visited[ep_id] = true;

        /// @brief Branch and Bound Algorithm
        float low_bound = dist;
        while (candidate_set.size())
        {
          auto curr_el_pair = candidate_set.top();
          if (-curr_el_pair.first > low_bound && top_candidates.size() == ef)
            break;
          candidate_set.pop();
          id_t curr_node_id = curr_el_pair.second;
          std::unique_lock<std::mutex> lock(*link_list_locks_[curr_node_id]);
          const auto& neighbors = link_lists_[curr_node_id][level];

          for (id_t neighbor_id: neighbors)
          {
            if (mass_visited[neighbor_id] == false)
            {
              mass_visited[neighbor_id] = true;

              float dist = distance(data_point, data_memory_[neighbor_id], D_);
              comparison++;

              /// @brief If neighbor is closer than farest vector in top result, and result.size still less than ef
              if (top_candidates.top().first > dist || top_candidates.size() < ef)
              {
                candidate_set.emplace(-dist, neighbor_id);
                top_candidates.emplace(dist, neighbor_id);

                if (top_candidates.size() > ef) // give up farest result so far
                  top_candidates.pop();

                if (top_candidates.size())
                  low_bound = top_candidates.top().first;
              }
            }
          }
        }
        comparison_.fetch_add(comparison);
        return top_candidates;
      }

        /// Codes below are for data collection
        std::priority_queue<std::pair<float, id_t>> SearchGetData(const data_t *query_data, size_t k, size_t ef, id_t qid, int data_type)
        {
            assert(ef >= k && "ef > k!");

            if (cur_element_count_ == 0)
                return std::priority_queue<std::pair<float, id_t>>();

            std::vector<data_t> dists, degrees;
            size_t num_updates = 0, num_lookback = 0;
            size_t comparison = 0;
            id_t cur_obj = enterpoint_node_;
            float cur_dist = distance(query_data, data_memory_[enterpoint_node_], D_);
            float ori_dist = cur_dist;
            comparison++;

            for (int lev = element_levels_[enterpoint_node_]; lev > 0; lev--)
            {
                bool changed = true;
                while (changed)
                {
                    changed = false;
                    const auto& neighbors = link_lists_[cur_obj][lev];
                    size_t num_neighbors = neighbors.size();

                    for (size_t i = 0; i < num_neighbors; i++)
                    {
                        id_t cand = neighbors[i];
                        float d = distance(query_data, data_memory_[cand], D_);
                        dists.emplace_back(d);
                        if (d < cur_dist)
                        {
                            cur_dist = d;
                            cur_obj = cand;
                            changed = true;
                            num_updates++;
                        }
                        if (d > ori_dist)
                        {
                            num_lookback++;
                        }
                    }
                    comparison += num_neighbors;
                    degrees.emplace_back(num_neighbors);
                }
            }

            while (dists.size() < num_check) {
                dists.emplace_back(0.);
            }
            std::partial_sort(dists.begin(), dists.begin() + num_check, dists.end());
            dists.resize(num_check);

            while (degrees.size() < 10) {
                degrees.emplace_back(0.);
            }
            std::partial_sort(degrees.begin(), degrees.begin() + 10, degrees.end(), std::greater<data_t>());
            degrees.resize(10);

            auto &vec_feats_hnns = data_type == 2 ? train_feats_nn[qid] : test_feats_nn[qid];

            for (int d = 0; d < D_; ++d) {
                vec_feats_hnns.emplace_back(query_data[d]);
            }

            for (int i = 0; i < dists.size(); ++i) {
                vec_feats_hnns.emplace_back(dists[i]);
            }
            for (int i = 0; i < degrees.size(); ++i) {
                vec_feats_hnns.emplace_back(degrees[i]);
            }
            vec_feats_hnns.emplace_back(num_updates);
            vec_feats_hnns.emplace_back(num_lookback);
            vec_feats_hnns.emplace_back(comparison);

            auto top_candidates = SearchBaseLayerGetData(cur_obj, query_data, 0, ef, qid, data_type);
            while (top_candidates.size() > k)
            {
                top_candidates.pop();
            }

            comparison_.fetch_add(comparison);

            return top_candidates;
        }

        void SearchGetData(const std::vector<std::vector<data_t>> &queries, size_t k, size_t ef, std::vector<std::vector<id_t>> &vids, std::vector<std::vector<float>> &dists, int data_type)
        {
            size_t nq = queries.size();
            vids.clear();
            dists.clear();
            vids.resize(nq);
            dists.resize(nq);

        #pragma omp parallel for schedule(dynamic, 1) num_threads(num_threads_)
            for (size_t i = 0; i < nq; i++)
            {
                const auto &query = queries[i];
                auto &vid = vids[i];
                auto &dist = dists[i];
                
                auto r = SearchGetData(query.data(), k, ef, i, data_type);
                vid.reserve(r.size());
                dist.reserve(r.size());
                while (r.size())
                {
                    const auto &te = r.top();
                    vid.emplace_back(te.second);
                    dist.emplace_back(te.first);
                    r.pop();
                }
                std::reverse(vid.begin(), vid.end());
                if (rand() % 10000 < 1) {
                    std::cout << "SearchGetData: " << i << " / " << nq << std::endl;
                }
            }
        }
      
        // 1: for query data (test set); 2: for train data (train set)
        std::priority_queue<std::pair<float, id_t>> SearchBaseLayerGetData(id_t ep_id, const data_t *data_point, int level, size_t ef, id_t qid, int data_type)
        {
            size_t comparison = 0;
            std::vector<bool> mass_visited(cur_element_count_, false);
            std::priority_queue<std::pair<float, id_t>> top_candidates;
            std::priority_queue<std::pair<float, id_t>> candidate_set;
            float dist = distance(data_point, data_memory_[ep_id], D_);
            comparison++;
            top_candidates.emplace(dist, ep_id); // max heap
            candidate_set.emplace(-dist, ep_id); // min heap
            mass_visited[ep_id] = true;

            /// @brief Branch and Bound Algorithm
            float low_bound = dist;
            float dist_start = dist;    // For SIGMOD20 lightgbm

            bool is_checked = false;

            auto &vec_label = data_type == 2 ? train_label[qid] : test_label[qid];
            auto &vec_feats_hnns = data_type == 2 ? train_feats_nn[qid] : test_feats_nn[qid];
            auto &vec_feats_lgb = data_type == 2 ? train_feats_lgb[qid] : test_feats_lgb[qid];
            const auto &gt = data_type == 2 ? train_gt : test_gt;


            while (candidate_set.size())
            {
                auto curr_el_pair = candidate_set.top();
                if (-curr_el_pair.first > low_bound && top_candidates.size() == ef)
                    break;
                candidate_set.pop();
                id_t curr_node_id = curr_el_pair.second;
                std::unique_lock<std::mutex> lock(*link_list_locks_[curr_node_id]);
                const auto& neighbors = link_lists_[curr_node_id][level];


                for (id_t neighbor_id: neighbors)
                {
                    if (mass_visited[neighbor_id] == false)
                    {
                        mass_visited[neighbor_id] = true;

                        float dist = distance(data_point, data_memory_[neighbor_id], D_);
                        comparison++;

                        if (comparison == check_stamp) 
                        {
                            is_checked = true;
                            auto top_candidates_backup = top_candidates;
                            std::vector<std::pair<float, id_t>> check_candidates;

                            while (top_candidates_backup.size()) {
                                auto curr_el_pair = top_candidates_backup.top();
                                top_candidates_backup.pop();
                                check_candidates.emplace_back(curr_el_pair);
                            }
                            std::reverse(check_candidates.begin(), check_candidates.end());
                            check_candidates.resize(num_check);

                            for (int d = 0; d < D_; ++d) {
                                vec_feats_lgb.emplace_back(data_point[d]);
                            }

                            vec_feats_lgb.emplace_back(-dist_start);
                            vec_feats_lgb.emplace_back(-check_candidates[0].first);
                            vec_feats_lgb.emplace_back(-check_candidates[9].first);
                            vec_feats_lgb.emplace_back(check_candidates[0].first / dist_start);
                            vec_feats_lgb.emplace_back(check_candidates[9].first / dist_start);

                        }

                        /// @brief If neighbor is closer than farest vector in top result, and result.size still less than ef
                        if (top_candidates.top().first > dist || top_candidates.size() < ef)
                        {
                            candidate_set.emplace(-dist, neighbor_id);
                            top_candidates.emplace(dist, neighbor_id);

                            // give up farest result so far
                            if (top_candidates.size() > ef) {
                                top_candidates.pop();
                            }
                            

                            if (top_candidates.size()) {
                                low_bound = top_candidates.top().first;
                            }
                        }
                    }
                }
            }

            auto top_candidates_backup = top_candidates;
            std::vector<id_t> knns;
            while (top_candidates_backup.size()) {
                auto curr_el_pair = top_candidates_backup.top();
                top_candidates_backup.pop();
                knns.emplace_back(curr_el_pair.second);
            }
            std::reverse(knns.begin(), knns.end());
            knns.resize(recall_at_k);

            auto recall_cnt = utils::GetRecallCount(recall_at_k, dimension_gt, gt, knns, qid);
            vec_label.emplace_back(recall_cnt);
            vec_label.emplace_back(comparison);

            comparison_.fetch_add(comparison);
            if (!is_checked) {
                auto top_candidates_backup = top_candidates;
                std::vector<std::pair<float, id_t>> check_candidates;

                while (top_candidates_backup.size()) {
                    auto curr_el_pair = top_candidates_backup.top();
                    top_candidates_backup.pop();
                    check_candidates.emplace_back(curr_el_pair);
                }
                std::reverse(check_candidates.begin(), check_candidates.end());
                check_candidates.resize(num_check);

                for (int d = 0; d < D_; ++d) {
                    vec_feats_lgb.emplace_back(data_point[d]);
                }

                vec_feats_lgb.emplace_back(-dist_start);
                vec_feats_lgb.emplace_back(-check_candidates[0].first);
                vec_feats_lgb.emplace_back(-check_candidates[9].first);
                vec_feats_lgb.emplace_back(check_candidates[0].first / dist_start);
                vec_feats_lgb.emplace_back(check_candidates[9].first / dist_start);
            }
            return top_candidates;
        }

        void SearchHNNS(const std::vector<std::vector<data_t>> &queries, size_t k, size_t ef, std::vector<std::vector<id_t>> &vids, std::vector<std::vector<float>> &dists, 
        std::vector<id_t> &qids, int data_type)
        {
            size_t nq = queries.size();
            vids.clear();
            dists.clear();
            vids.resize(nq);
            dists.resize(nq);

            if (data_type == 0) {
            #pragma omp parallel for schedule(dynamic, 1) num_threads(num_threads_)
                for (size_t i = 0; i < nq; i++)
                {
                    const auto &query = queries[i];
                    auto &score = dists[i];
                    auto r = SearchHNNS(query.data(), k, ef, qids[i], 0);
                    score.emplace_back(r.top().first);
                }
            } else {
            #pragma omp parallel for schedule(dynamic, 1) num_threads(num_threads_)
                for (size_t qid = 0; qid < nq; qid++)
                {
                    const auto &query = queries[qid];
                    auto &vid = vids[qid];
                    auto &dist = dists[qid];
                    
                    auto r = SearchHNNS(query.data(), k, ef, qids[qid], data_type);
                    vid.reserve(r.size());
                    dist.reserve(r.size());
                    while (r.size())
                    {
                        const auto &te = r.top();
                        vid.emplace_back(te.second);
                        dist.emplace_back(te.first);
                        r.pop();
                    }
                    std::reverse(vid.begin(), vid.end());
                    if (rand() % 10000 < 1) {
                        std::cout << "SearchHNNS: " << qid << " / " << nq << std::endl;
                    }

                }
            }
        }

        std::priority_queue<std::pair<float, id_t>> SearchHNNS(const data_t *query_data, size_t k, size_t ef, id_t qid, int data_type)
        {
            assert(ef >= k && "ef > k!");

            if (cur_element_count_ == 0)
                return std::priority_queue<std::pair<float, id_t>>();

            std::vector<data_t> dists;

            size_t comparison = 0;
            id_t cur_obj = enterpoint_node_;
            if (data_type == 0 || !test_inter_results[qid].ready) {
                // float cur_dist = distance(query_data, data_memory_[enterpoint_node_], D_);
                // comparison++;

                // for (int lev = element_levels_[enterpoint_node_]; lev > 0; lev--)
                // {
                //     bool changed = true;
                //     while (changed)
                //     {
                //         changed = false;
                //         const auto& neighbors = link_lists_[cur_obj][lev];
                //         size_t num_neighbors = neighbors.size();

                //         for (size_t i = 0; i < num_neighbors; i++)
                //         {
                //             id_t cand = neighbors[i];
                //             float d = distance(query_data, data_memory_[cand], D_);
                //             if (d < cur_dist)
                //             {
                //                 cur_dist = d;
                //                 cur_obj = cand;
                //                 changed = true;
                //             }
                //         }
                //         comparison += num_neighbors;
                //     }
                // }
                // test_inter_results[qid].NDC = comparison;
                std::vector<data_t> dists, degrees;
                size_t num_updates = 0, num_lookback = 0;

                size_t comparison = 0;
                id_t cur_obj = enterpoint_node_;
                float cur_dist = distance(query_data, data_memory_[enterpoint_node_], D_);
                float ori_dist = cur_dist;
                comparison++;

                for (int lev = element_levels_[enterpoint_node_]; lev > 0; lev--)
                {
                    bool changed = true;
                    while (changed)
                    {
                        changed = false;
                        const auto& neighbors = link_lists_[cur_obj][lev];
                        size_t num_neighbors = neighbors.size();

                        for (size_t i = 0; i < num_neighbors; i++)
                        {
                            id_t cand = neighbors[i];
                            float d = distance(query_data, data_memory_[cand], D_);
                            dists.emplace_back(d);
                            if (d < cur_dist)
                            {
                                cur_dist = d;
                                cur_obj = cand;
                                changed = true;
                                num_updates++;
                            }
                            if (d > ori_dist)
                            {
                                num_lookback++;
                            }
                        }
                        comparison += num_neighbors;
                        degrees.emplace_back(num_neighbors);
                    }
                }

                while (dists.size() < num_check) {
                    dists.emplace_back(0.);
                }
                std::partial_sort(dists.begin(), dists.begin() + num_check, dists.end());
                dists.resize(num_check);

                while (degrees.size() < 10) {
                    degrees.emplace_back(0.);
                }
                std::partial_sort(degrees.begin(), degrees.begin() + 10, degrees.end(), std::greater<data_t>());
                degrees.resize(10);

                auto &vec_feats_hnns = data_type == 2 ? train_feats_nn[qid] : test_feats_nn[qid];
                vec_feats_hnns.clear();

                for (int d = 0; d < D_; ++d) {
                    vec_feats_hnns.emplace_back(query_data[d]);
                }

                for (int i = 0; i < dists.size(); ++i) {
                    vec_feats_hnns.emplace_back(dists[i]);
                }
                for (int i = 0; i < degrees.size(); ++i) {
                    vec_feats_hnns.emplace_back(degrees[i]);
                }
                vec_feats_hnns.emplace_back(num_updates);
                vec_feats_hnns.emplace_back(num_lookback);
                vec_feats_hnns.emplace_back(comparison);
                // assert (vec_feats_hnns.size() == num_check + D_ + 10 + 3);

                int64_t out_len;
                double out_result = 0.;
                LGBM_BoosterPredictForMat(handle, vec_feats_hnns.data(), C_API_DTYPE_FLOAT32, 
                    1, vec_feats_hnns.size(), 1, C_API_PREDICT_NORMAL, 0, -1, "", &out_len, &out_result);
                std::priority_queue<std::pair<float, id_t>> queue_for_score;
                queue_for_score.emplace(out_result, 0);
                test_inter_results[qid].ready = true;
                test_inter_results[qid].enterpoint_node = cur_obj;
                return queue_for_score;
            } else {
                cur_obj = test_inter_results[qid].enterpoint_node;
                test_inter_results[qid].ready = false;
            }

            auto top_candidates = SearchBaseLayerHNNS(cur_obj, query_data, 0, ef, qid, data_type);

            while (top_candidates.size() > k)
            {
                top_candidates.pop();
            }

            comparison_.fetch_add(comparison);

            return top_candidates;
        }

        // 0: for 1st stage LightGBM inference; 1: for 2nd stage search
        std::priority_queue<std::pair<float, id_t>> SearchBaseLayerHNNS(id_t ep_id, const data_t *data_point, int level, size_t ef, id_t qid, int data_type)
        {
            size_t comparison = 0;
            size_t num_lookback = 0;
            size_t num_pop = 0;
            size_t num_lb_update = 0;
            auto &vec_feats_hnns = test_feats_nn[qid];
            if (data_type == 0) vec_feats_hnns.clear();

            // bool first_stage_ready = test_inter_results[qid].ready;
            float low_bound;
            float dist_start;    // For SIGMOD20 lightgbm
            std::vector<bool> mass_visited(cur_element_count_, false);
            std::vector<id_t> visited;
            std::priority_queue<std::pair<float, id_t>> top_candidates;
            std::priority_queue<std::pair<float, id_t>> candidate_set;
            // if (first_stage_ready) {
            // // if (false) {
            //     std::swap(test_inter_results[qid].top_candidates, top_candidates);
            //     std::swap(test_inter_results[qid].candidate_set, candidate_set);
            //     std::swap(test_inter_results[qid].visited, visited);
            //     std::swap(test_inter_results[qid].low_bound, low_bound);
            //     for (auto &v: visited) 
            //         mass_visited[v] = true;
            // } else {
                float dist = distance(data_point, data_memory_[ep_id], D_);
                comparison++;
                top_candidates.emplace(dist, ep_id); // max heap
                candidate_set.emplace(-dist, ep_id); // min heap

                mass_visited[ep_id] = true;
                visited.emplace_back(ep_id);

                low_bound = dist;
                dist_start = dist;    // For SIGMOD20 lightgbm
            // }

            bool is_checked = false;

            while (candidate_set.size())
            {
                auto curr_el_pair = candidate_set.top();
                if (-curr_el_pair.first > low_bound && top_candidates.size() == ef)
                    break;
                candidate_set.pop();
                id_t curr_node_id = curr_el_pair.second;
                std::unique_lock<std::mutex> lock(*link_list_locks_[curr_node_id]);
                const auto& neighbors = link_lists_[curr_node_id][level];

                for (id_t neighbor_id: neighbors)
                {
                    if (mass_visited[neighbor_id] == false)
                    {
                        mass_visited[neighbor_id] = true;
                        visited.emplace_back(neighbor_id);

                        float dist = distance(data_point, data_memory_[neighbor_id], D_);
                        comparison++;
                        if (!is_checked && dist > dist_start) {
                            num_lookback++;
                        }

                        if (data_type == 0 && comparison == check_stamp) 
                        {
                            is_checked = true;
                            auto top_candidates_backup = top_candidates;
                            std::vector<std::pair<float, id_t>> check_candidates;

                            while (top_candidates_backup.size()) {
                                auto curr_el_pair = top_candidates_backup.top();
                                top_candidates_backup.pop();
                                check_candidates.emplace_back(curr_el_pair);
                            }
                            std::reverse(check_candidates.begin(), check_candidates.end());
                            check_candidates.resize(num_check);

                            for (int d = 0; d < D_; ++d) {
                                vec_feats_hnns.emplace_back(data_point[d]);
                            }

                            for (int i = 0; i < check_candidates.size(); ++i) {
                                vec_feats_hnns.emplace_back(check_candidates[i].first);
                            }
                            vec_feats_hnns.emplace_back(num_lookback);
                            vec_feats_hnns.emplace_back(num_pop);
                            vec_feats_hnns.emplace_back(num_lb_update);
                        }

                        /// @brief If neighbor is closer than farest vector in top result, and result.size still less than ef
                        if (top_candidates.top().first > dist || top_candidates.size() < ef)
                        {
                            candidate_set.emplace(-dist, neighbor_id);
                            top_candidates.emplace(dist, neighbor_id);

                            // give up farest result so far
                            if (top_candidates.size() > ef) {
                                top_candidates.pop();
                                num_pop++;
                            }
                            

                            if (top_candidates.size()) {
                                low_bound = top_candidates.top().first;
                                num_lb_update++;
                            }
                        }
                    }
                }

                if (is_checked && data_type == 0) {
                    int64_t out_len;
                    double out_result = 0.;
                    LGBM_BoosterPredictForMat(handle, vec_feats_hnns.data(), C_API_DTYPE_FLOAT32, 
                        1, vec_feats_hnns.size(), 1, C_API_PREDICT_NORMAL, 0, -1, "", &out_len, &out_result);
                    
                    comparison_.fetch_add(comparison);
                    std::priority_queue<std::pair<float, id_t>> queue_for_score;
                    queue_for_score.emplace(out_result, 0);

                    // std::swap(test_inter_results[qid].top_candidates, top_candidates);
                    // std::swap(test_inter_results[qid].candidate_set, candidate_set);
                    // std::swap(test_inter_results[qid].visited, visited);
                    // std::swap(test_inter_results[qid].low_bound, low_bound);
                    // test_inter_results[qid].ready = true;

                    return queue_for_score;
                }
            }

            comparison_.fetch_add(comparison);
            return top_candidates;
        }
      
        void SearchEarlyStop(const std::vector<std::vector<data_t>> &queries, size_t k, size_t ef, std::vector<std::vector<id_t>> &vids, std::vector<std::vector<float>> &dists, 
        std::vector<id_t> &qids, int data_type)
        {
            size_t nq = queries.size();
            vids.clear();
            dists.clear();
            vids.resize(nq);
            dists.resize(nq);

            if (data_type == 0) {
            #pragma omp parallel for schedule(dynamic, 1) num_threads(num_threads_)
                for (size_t i = 0; i < nq; i++)
                {
                    const auto &query = queries[i];
                    auto &score = dists[i];
                    auto r = SearchEarlyStop(query.data(), k, ef, qids[i], 0);
                    score.emplace_back(r.top().first);
                }
            } else {
            #pragma omp parallel for schedule(dynamic, 1) num_threads(num_threads_)
                for (size_t qid = 0; qid < nq; qid++)
                {
                    const auto &query = queries[qid];
                    auto &vid = vids[qid];
                    auto &dist = dists[qid];
                    
                    auto r = SearchEarlyStop(query.data(), k, ef, qids[qid], data_type);
                    vid.reserve(r.size());
                    dist.reserve(r.size());
                    while (r.size())
                    {
                        const auto &te = r.top();
                        vid.emplace_back(te.second);
                        dist.emplace_back(te.first);
                        r.pop();
                    }
                    std::reverse(vid.begin(), vid.end());
                    if (rand() % 10000 < 1) {
                        std::cout << "SearchEarlyStop: " << qid << " / " << nq << std::endl;
                    }

                }
            }
        }

        std::priority_queue<std::pair<float, id_t>> SearchEarlyStop(const data_t *query_data, size_t k, size_t ef, id_t qid, int data_type)
        {
            assert(ef >= k && "ef > k!");

            if (cur_element_count_ == 0)
                return std::priority_queue<std::pair<float, id_t>>();

            size_t comparison = 0;
            id_t cur_obj = enterpoint_node_;
            float cur_dist = distance(query_data, data_memory_[enterpoint_node_], D_);
            comparison++;

            for (int lev = element_levels_[enterpoint_node_]; lev > 0; lev--)
            {
                bool changed = true;
                while (changed)
                {
                    changed = false;
                    const auto& neighbors = link_lists_[cur_obj][lev];
                    size_t num_neighbors = neighbors.size();

                    for (size_t i = 0; i < num_neighbors; i++)
                    {
                        id_t cand = neighbors[i];
                        float d = distance(query_data, data_memory_[cand], D_);
                        if (d < cur_dist)
                        {
                            cur_dist = d;
                            cur_obj = cand;
                            changed = true;
                        }
                    }
                    comparison += num_neighbors;
                }
            }

            auto top_candidates = SearchBaseLayerEarlyStop(cur_obj, query_data, 0, ef, qid, data_type);

            while (top_candidates.size() > k)
            {
                top_candidates.pop();
            }

            comparison_.fetch_add(comparison);

            return top_candidates;
        }

        // 0: for 1st stage LightGBM inference; 1: for 2nd stage search
        std::priority_queue<std::pair<float, id_t>> SearchBaseLayerEarlyStop(id_t ep_id, const data_t *data_point, int level, size_t ef, id_t qid, int data_type)
        {
            size_t comparison = 0;
            auto &vec_feats_lgb = test_feats_lgb[qid];
            if (data_type == 0) vec_feats_lgb.clear();

            // bool first_stage_ready = test_inter_results[qid].ready;
            float low_bound;
            float dist_start;    // For SIGMOD20 lightgbm
            std::vector<bool> mass_visited(cur_element_count_, false);
            std::vector<id_t> visited;
            std::priority_queue<std::pair<float, id_t>> top_candidates;
            std::priority_queue<std::pair<float, id_t>> candidate_set;
            // if (first_stage_ready) {
            // // if (false) {
            //     std::swap(test_inter_results[qid].top_candidates, top_candidates);
            //     std::swap(test_inter_results[qid].candidate_set, candidate_set);
            //     std::swap(test_inter_results[qid].visited, visited);
            //     std::swap(test_inter_results[qid].low_bound, low_bound);
            //     for (auto &v: visited) 
            //         mass_visited[v] = true;
            // } else {
                float dist = distance(data_point, data_memory_[ep_id], D_);
                comparison++;
                top_candidates.emplace(dist, ep_id); // max heap
                candidate_set.emplace(-dist, ep_id); // min heap

                mass_visited[ep_id] = true;
                visited.emplace_back(ep_id);

                low_bound = dist;
                dist_start = dist;    // For SIGMOD20 lightgbm
            // }

            bool is_checked = false;

            while (candidate_set.size())
            {
                auto curr_el_pair = candidate_set.top();
                if (-curr_el_pair.first > low_bound && top_candidates.size() == ef)
                    break;
                candidate_set.pop();
                id_t curr_node_id = curr_el_pair.second;
                std::unique_lock<std::mutex> lock(*link_list_locks_[curr_node_id]);
                const auto& neighbors = link_lists_[curr_node_id][level];

                for (id_t neighbor_id: neighbors)
                {
                    if (mass_visited[neighbor_id] == false)
                    {
                        mass_visited[neighbor_id] = true;
                        visited.emplace_back(neighbor_id);

                        float dist = distance(data_point, data_memory_[neighbor_id], D_);
                        comparison++;

                        if (data_type == 0 && comparison == check_stamp) 
                        {
                            is_checked = true;
                            auto top_candidates_backup = top_candidates;
                            std::vector<std::pair<float, id_t>> check_candidates;

                            while (top_candidates_backup.size()) {
                                auto curr_el_pair = top_candidates_backup.top();
                                top_candidates_backup.pop();
                                check_candidates.emplace_back(curr_el_pair);
                            }
                            std::reverse(check_candidates.begin(), check_candidates.end());
                            check_candidates.resize(num_check);

                            for (int d = 0; d < D_; ++d) {
                                vec_feats_lgb.emplace_back(data_point[d]);
                            }

                            vec_feats_lgb.emplace_back(-dist_start);
                            vec_feats_lgb.emplace_back(-check_candidates[0].first);
                            vec_feats_lgb.emplace_back(-check_candidates[9].first);
                            vec_feats_lgb.emplace_back(check_candidates[0].first / dist_start);
                            vec_feats_lgb.emplace_back(check_candidates[9].first / dist_start);
                        }

                        /// @brief If neighbor is closer than farest vector in top result, and result.size still less than ef
                        if (top_candidates.top().first > dist || top_candidates.size() < ef)
                        {
                            candidate_set.emplace(-dist, neighbor_id);
                            top_candidates.emplace(dist, neighbor_id);

                            // give up farest result so far
                            if (top_candidates.size() > ef) {
                                top_candidates.pop();
                            }
                            

                            if (top_candidates.size()) {
                                low_bound = top_candidates.top().first;
                            }
                        }
                    }
                }

                if (!is_checked) {
                    is_checked = true;
                    auto top_candidates_backup = top_candidates;
                    std::vector<std::pair<float, id_t>> check_candidates;

                    while (top_candidates_backup.size()) {
                        auto curr_el_pair = top_candidates_backup.top();
                        top_candidates_backup.pop();
                        check_candidates.emplace_back(curr_el_pair);
                    }
                    std::reverse(check_candidates.begin(), check_candidates.end());
                    check_candidates.resize(num_check);

                    for (int d = 0; d < D_; ++d) {
                        vec_feats_lgb.emplace_back(data_point[d]);
                    }

                    vec_feats_lgb.emplace_back(-dist_start);
                    vec_feats_lgb.emplace_back(-check_candidates[0].first);
                    vec_feats_lgb.emplace_back(-check_candidates[9].first);
                    vec_feats_lgb.emplace_back(check_candidates[0].first / dist_start);
                    vec_feats_lgb.emplace_back(check_candidates[9].first / dist_start);
                }

                if (is_checked && data_type == 0) {
                    int64_t out_len;
                    double out_result = 0.;
                    LGBM_BoosterPredictForMat(handle, vec_feats_lgb.data(), C_API_DTYPE_FLOAT32, 
                        1, vec_feats_lgb.size(), 1, C_API_PREDICT_NORMAL, 0, -1, "", &out_len, &out_result);
                    
                    comparison_.fetch_add(comparison);
                    std::priority_queue<std::pair<float, id_t>> queue_for_score;
                    queue_for_score.emplace(out_result, 0);

                    // std::swap(test_inter_results[qid].top_candidates, top_candidates);
                    // std::swap(test_inter_results[qid].candidate_set, candidate_set);
                    // std::swap(test_inter_results[qid].visited, visited);
                    // std::swap(test_inter_results[qid].low_bound, low_bound);
                    // test_inter_results[qid].ready = true;

                    return queue_for_score;
                }
            }

            comparison_.fetch_add(comparison);
            return top_candidates;
        }
      
        void SaveData(std::string data_prefix, size_t ef_search, bool is_OOD=true)
        {
            size_t recall_threshold = dataset_threshold[dataset];
            recall_threshold = 400;
            std::cout << "[HNSW] recall_threshold: " << recall_threshold << std::endl;
            std::cout << "[HNSW] check_stamp: " << check_stamp << std::endl;
            this->prefix = dataset + "."
                "M_" + std::to_string(Mmax_) + "."
                "efc_" + std::to_string(ef_construction_) + "."
                "efs_" + std::to_string(ef_search) + "."
                "ck_ts_" + std::to_string(check_stamp) + "."
                "ncheck_" + std::to_string(num_check) + "."
                "recall@" + std::to_string(recall_at_k);
            
            if (!is_OOD) {
                this->prefix += ".IID";
            }

            size_t train_label_postive = 0, test_label_postive = 0;
            for (int i = 0; i < train_label.size(); ++i) {
                assert(train_label[i].size() == 2);
                if (train_label[i][0] < recall_threshold) continue;
                train_label_postive += 1;
            }
            std::cout << "train_label_postive: " << train_label_postive << std::endl;
            for (int i = 0; i < test_label.size(); ++i) {
                assert(test_label[i].size() == 2);
                if (test_label[i][0] < recall_threshold) continue;
                test_label_postive += 1;
            }
            std::cout << "test_label_postive: " << test_label_postive << std::endl;

            std::cout << "Saving to: " << data_prefix + this->prefix << std::endl;
            utils::WriteToFile<int>(utils::Flatten(test_label), {test_label.size(), test_label[0].size()}, data_prefix + this->prefix + ".test_label.ivecs");
            utils::WriteToFile<float>(utils::Flatten(test_feats_nn), {test_feats_nn.size(), test_feats_nn[0].size()}, data_prefix + this->prefix + ".test_feats_nn.fvecs");
            utils::WriteToFile<float>(utils::Flatten(test_feats_lgb), {test_feats_lgb.size(), test_feats_lgb[0].size()}, data_prefix + this->prefix + ".test_feats_lgb.fvecs");

            std::vector<std::vector<int>> train_label_valid;
            std::vector<std::vector<float>> train_feats_nn_valid, train_feats_lgb_valid;

            for (int i = 0; i < train_label.size(); ++i) {
                train_label_valid.emplace_back(train_label[i]);
                train_feats_nn_valid.emplace_back(train_feats_nn[i]);
                train_feats_lgb_valid.emplace_back(train_feats_lgb[i]);
            }
            utils::WriteToFile<int>(utils::Flatten(train_label_valid), {train_label_valid.size(), train_label_valid[0].size()}, data_prefix + this->prefix + ".train_label.ivecs");
            utils::WriteToFile<float>(utils::Flatten(train_feats_nn_valid), {train_feats_nn_valid.size(), train_feats_nn_valid[0].size()}, data_prefix + this->prefix + ".train_feats_nn.fvecs");
            utils::WriteToFile<float>(utils::Flatten(train_feats_lgb_valid), {train_feats_lgb_valid.size(), train_feats_lgb_valid[0].size()}, data_prefix + this->prefix + ".train_feats_lgb.fvecs");
        }

        // void Reset() 
        // {
        //     test_inter_results.clear();
        //     test_inter_results.resize(test_label.size(), IntermediateResult(check_stamp));
        // }
        void LoadGT(std::vector<id_t>& query_gt, std::vector<id_t>& learn_gt)
        {
            num_test = query_gt.size() / 1000;
            num_train = learn_gt.size() / 1000;
            
            test_gt = query_gt;
            train_gt = learn_gt;
            
            std::cout << "num_test: " << num_test << std::endl;
            std::cout << "num_train: " << num_train << std::endl;

            train_feats_nn.resize(num_train);
            train_feats_lgb.resize(num_train);
            train_label.resize(num_train);
            test_feats_nn.resize(num_test);
            test_feats_lgb.resize(num_test);
            test_label.resize(num_test);
            test_inter_results.resize(num_test, IntermediateResult(check_stamp));
        }

        void LoadLightGBM(BoosterHandle handle)
        {
            this->handle = handle;
            lgb_ready = true;
        }
    };

  }; // namespace graph

}; // namespace index
