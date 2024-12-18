#include <string>
#include <vector>
#include "graph/hnsw.hpp"
#include "utils/resize.hpp"
#include "utils/timer.hpp"
#include "utils/recall.hpp"
#include "utils/dataloader.hpp"
#include "distance.hpp"

using data_t = float;
using id_t = uint32_t;
using namespace std;

std::string prefix = "/home/zhengweiguo/liuchengjun/";
std::string idx_prefix = "/data/disk1/liuchengjun/HNNS/checkpoint/";
float (*metric)(const data_t *, const data_t *, size_t) = nullptr;

int main(int argc, char** argv) 
{
    std::vector<data_t> base_vectors, query_vectors, learn_vectors;
    std::vector<id_t> query_gt_vectors, learn_gt_vectors;
    std::string base_name = std::string(argv[1]);
    std::string query_name = std::string(argv[2]);
    size_t M = std::stol(argv[3]);
    size_t efconsturct = std::stol(argv[4]);
    size_t k = std::stol(argv[5]);
    utils::DataLoader data_loader(base_name, query_name);
    utils::BaseQueryGtConfig cfg;
    std::tie(base_vectors, query_vectors, query_gt_vectors, cfg)
         = data_loader.load();
    if (cfg.metric == 0) {
        metric = InnerProduct;
        std::cout << "[Metric] InnerProduct" << std::endl;
    } else {
        metric = L2;
        std::cout << "[Metric] L2" << std::endl;
    }
    auto nest_test_vectors = utils::Nest(std::move(query_vectors), cfg.num_query, cfg.dim_query);

    // // query mask
    // cfg.query_gt_path += ".mask1";
    // for (auto &v : nest_test_vectors) {
    //     for (size_t d = 0; d < cfg.dim_query / 2; ++d) {
    //         v[d] = 0.;
    //     }
    // }
    
    base_vectors.resize(cfg.num_base * cfg.dim_base / 1);
    cfg.num_base = base_vectors.size() / cfg.dim_base;

    nest_test_vectors.resize(cfg.num_query / 1);
    cfg.num_query = nest_test_vectors.size();

    cout << "Load Data Done!" << endl;

    size_t check_stamp = 2000;
    size_t num_check = 100;

    utils::Timer build_timer, query_timer;
    size_t efq = efconsturct;
    // size_t efq = 10000;
    
    std::string index_path = 
        // "../index/" + base_name + "."
        "/data/disk1/liuchengjun/HNNS/index/" + base_name + "."
        "M_" + to_string(M) + "." 
        "efc_" + to_string(efconsturct) + ".hnsw";

    std::cout << "base_name: " << base_name << std::endl;
    std::cout << "efSearch: " << efq << std::endl;
    std::cout << "efConstruct: " << efconsturct << std::endl;
    std::cout << "M: " << M << std::endl;
    std::cout << "index_path: " << index_path << std::endl;

    auto hnsw = std::make_unique<anns::graph::HNSW<data_t>> (
        base_vectors, index_path, base_name,
        k, check_stamp, metric);
    hnsw->SetNumThreads(48);

    build_timer.Start();
    // hnsw->BuildIndex(base_vectors);
    // hnsw->Save(index_path);
    build_timer.Stop();

    std::vector<std::vector<id_t>> knn;
    std::vector<std::vector<data_t>> dists;
    hnsw->GetComparisonAndClear();
    std::cout << "Build Time: " << build_timer.GetTime() << std::endl;
    
    query_timer.Reset();
    query_timer.Start();
    hnsw->Search(nest_test_vectors, k, efq, knn, dists);
    query_timer.Stop();
    std::cout << "[Query][HNSW] Params: " << "M: " << M << ", efSearch: " << efq << std::endl;
    std::cout << "[Query][HNSW] Using GT from file: " << cfg.query_gt_path << std::endl;
    std::cout << "[Query][HNSW] Search time: " << query_timer.GetTime() << std::endl;
    // std::cout << "[Query][HNSW] Recall@" << k << ": " << utils::GetRecall(k, cfg.dim_query_gt, query_gt_vectors, knn) << std::endl;
    for (int ck = 1; ck <= k; ck *= 10) {
        std::cout << "[Query][HNSW] Recall@" << ck << ": " << utils::GetRecall(ck, cfg.dim_query_gt, query_gt_vectors, knn) << std::endl;
    }
    std::cout << "[Query][HNSW] avg comparison: " << hnsw->GetComparisonAndClear() / (double)cfg.num_query << std::endl;

    return 0;
}

// ./hnsw_run imagenet 96 3000 1000 1000
// ./hnsw_run wikipedia 128 3000 1000 1000
// ./hnsw_run datacomp-image 32 1000 1000 1000
// ./hnsw_run deep100m 32 1000 1000 1000