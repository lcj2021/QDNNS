#include <string>
#include <vector>
#include "graph/hnsw.hpp"
#include "utils/dataloader.hpp"
#include "utils/resize.hpp"
#include "utils/timer.hpp"
#include "utils/recall.hpp"
#include "distance.hpp"

using data_t = float;
using id_t = uint32_t;
using namespace std;

std::string prefix = "/home/zhengweiguo/liuchengjun/";
std::string save_prefix = "/data/disk1/liuchengjun/HNNS/sample/";
float (*metric)(const data_t *, const data_t *, size_t) = nullptr;

int main(int argc, char** argv) 
{
    std::vector<data_t> base_vectors, query_vectors, learn_vectors;
    std::vector<id_t> query_gt_vectors, learn_gt_vectors;
    // std::string dataset = "gist1m";
    // std::string dataset = "imagenet";
    // std::string dataset = "wikipedia";
    std::string base_name = std::string(argv[1]);
    std::string query_name = std::string(argv[2]);
    size_t M = std::stol(argv[3]);
    size_t efq = std::stol(argv[4]);
    size_t check_stamp = std::stol(argv[5]);
    std::string base_vectors_path;
    std::string test_vectors_path;
    std::string test_gt_path;
    std::string learn_vectors_path;
    std::string learn_gt_path;
    utils::DataLoader data_loader(base_name, query_name);
    utils::BaseQueryLearnGtConfig cfg;
    std::tie(base_vectors, 
        query_vectors, query_gt_vectors, 
        learn_vectors, learn_gt_vectors,  cfg)
         = data_loader.load_with_learn();
    if (cfg.metric == 0) {
        metric = InnerProduct;
        std::cout << "[Metric] InnerProduct" << std::endl;
    } else {
        metric = L2;
        std::cout << "[Metric] L2" << std::endl;
    }
    
    auto nest_query_vectors = utils::Nest(std::move(query_vectors), cfg.num_query, cfg.dim_query);
    auto nest_learn_vectors = utils::Nest(std::move(learn_vectors), cfg.num_learn, cfg.dim_learn);

    cout << "Load Data Done!" << endl;

    size_t k = 1000;

    utils::Timer build_timer;
    utils::Timer query_timer;
    size_t ef_construction = 1000;
    std::string index_path = 
        // "../index/" + dataset + "."
        "/data/disk1/liuchengjun/HNNS/index/" + base_name + "."
        "M_" + to_string(M) + "." 
        "efc_" + to_string(ef_construction) + ".hnsw";
    std::cout << "dataset: " << base_name << std::endl;
    std::cout << "efSearch: " << efq << std::endl;
    std::cout << "efConstruct: " << ef_construction << std::endl;
    std::cout << "M: " << M << std::endl;

    auto hnsw = std::make_unique<anns::graph::HNSW<data_t>> (
        base_vectors, index_path, base_name,
        k, check_stamp, metric);
    hnsw->SetNumThreads(96);

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
    hnsw->SearchGetData(nest_query_vectors, k, efq, knn, dists, 1);
    query_timer.Stop();
    std::cout << "[Query][HNSW] Using GT from file: " << test_gt_path << std::endl;
    std::cout << "[Query][HNSW] Search time: " << query_timer.GetTime() << std::endl;
    std::cout << "[Query][HNSW] Recall@" << k << ": " << utils::GetRecall(k, cfg.dim_query_gt, query_gt_vectors, knn) << std::endl;
    std::cout << "[Query][HNSW] avg comparison: " << hnsw->GetComparisonAndClear() / (double)cfg.num_query << std::endl;

    query_timer.Reset();
    query_timer.Start();
    hnsw->SearchGetData(nest_learn_vectors, k, efq, knn, dists, 2);
    query_timer.Stop();
    std::cout << "[learn][HNSW] Using GT from file: " << learn_gt_path << std::endl;
    std::cout << "[learn][HNSW] Search time: " << query_timer.GetTime() << std::endl;
    for (int ck = 1; ck <= k; ck *= 10) {
        std::cout << "[learn][HNSW] Recall@" << ck << ": " << utils::GetRecall(ck, cfg.dim_learn_gt, learn_gt_vectors, knn) << std::endl;
    }
    // std::cout << "[learn][HNSW] Recall@" << k << ": " << utils::GetRecall(k, cfg.dim_learn_gt, learn_gt_vectors, knn) << std::endl;
    std::cout << "[learn][HNSW] avg comparison: " << hnsw->GetComparisonAndClear() / (double)cfg.num_learn << std::endl;
    hnsw->SaveData(save_prefix, efq);
    return 0;
}

// sudo ./hnsw_get_data imagenet.base imagenet.query 32 1000 1000
// sudo ./hnsw_get_data wikipedia.base wikipedia.query 32 1000 1000
// sudo ./hnsw_get_data deep100m 32 1000 1000
// sudo ./hnsw_get_data deep100m 32 1000 1000