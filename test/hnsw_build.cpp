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
float (*metric)(const data_t *, const data_t *, size_t) = nullptr;

int main(int argc, char** argv) 
{
    std::vector<data_t> base_vectors, queries_vectors;
    std::vector<id_t> gt_vectors;
    std::string base_name = std::string(argv[1]);
    std::string query_name = std::string(argv[2]);
    size_t M = std::stol(argv[3]);
    size_t ef_construction = std::stol(argv[4]);
    size_t k = std::stol(argv[5]);
    utils::DataLoader data_loader;
    utils::BaseQueryGtConfig cfg;
    std::tie(base_vectors, queries_vectors, gt_vectors, cfg)
         = data_loader.load(base_name, query_name);
    if (cfg.metric == 0) {
        metric = InnerProduct;
        std::cout << "[Metric] InnerProduct" << std::endl;
    } else {
        metric = L2;
        std::cout << "[Metric] L2" << std::endl;
    }
    auto nest_test_vectors = utils::Nest(std::move(queries_vectors), cfg.num_query, cfg.dim_query);

    size_t efq = ef_construction;
    size_t check_stamp = 2000;
    std::cout << "efSearch: " << efq << std::endl;

    utils::Timer build_timer, query_timer;
    std::string index_path = 
        // "../index/" + dataset + "."
        "/data/disk1/liuchengjun/HNNS/index/" + base_name + "."
        "M_" + to_string(M) + "." 
        "efc_" + to_string(ef_construction) + ".hnsw";
    std::cout << "base_name: " << base_name << std::endl;
    std::cout << "efSearch: " << efq << std::endl;
    std::cout << "efConstruct: " << ef_construction << std::endl;
    std::cout << "M: " << M << std::endl;

    auto hnsw = std::make_unique<anns::graph::HNSW<data_t>> (cfg.dim_base, cfg.num_base, M, ef_construction,
        base_name, k, check_stamp, metric
    );
    hnsw->SetNumThreads(12);

    build_timer.Start();
    hnsw->BuildIndex(base_vectors);
    hnsw->Save(index_path);
    build_timer.Stop();

    hnsw->check_stamp = check_stamp;
    std::cout << "check_stamp: " << check_stamp << std::endl;
    std::vector<std::vector<id_t>> knn;
    std::vector<std::vector<data_t>> dists;
    hnsw->GetComparisonAndClear();
    std::cout << "[Base][HNSW] Build Time: " << build_timer.GetTime() << std::endl;
    
    query_timer.Reset();
    query_timer.Start();
    hnsw->Search(nest_test_vectors, k, efq, knn, dists);
    query_timer.Stop();
    std::cout << "[Query][HNSW] Using GT from file: " << cfg.gt_path << std::endl;
    std::cout << "[Query][HNSW] Search time: " << query_timer.GetTime() << std::endl;
    std::cout << "[Query][HNSW] Recall@" << k << ": " << utils::GetRecall(k, cfg.dim_gt, gt_vectors, knn) << std::endl;
    std::cout << "[Query][HNSW] avg comparison: " << hnsw->GetComparisonAndClear() / (double)cfg.num_query << std::endl;
    return 0;
}

// sudo ./hnsw_build wikipedia base 96
// sudo ./hnsw_build wikipedia base 48
// sudo ./hnsw_build imagenet base 96
// sudo ./hnsw_build imagenet base 48