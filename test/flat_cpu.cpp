#include <algorithm>
#include "utils/binary_io.hpp"
#include "utils/resize.hpp"
#include "utils/timer.hpp"
#include "utils/recall.hpp"
#include "utils/dataloader.hpp"
#include "flat/IndexFlat.hpp"
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
    // std::string dataset = "gist1m";
    // std::string dataset = "imagenet";
    // std::string dataset = "wikipedia";
    // std::string dataset = "datacomp-image";
    // std::string dataset = "deep100m";
    std::string base_name = std::string(argv[1]);
    std::string query_name = std::string(argv[2]);
    utils::DataLoader data_loader;
    std::string gt_path;
    size_t num_base, dim_base, num_query, dim_query, num_gt, dim_gt;
    int metric_type = 0;
    utils::BaseQueryGtConfig cfg;
    std::tie(base_vectors, queries_vectors, gt_vectors, cfg)
         = data_loader.load(base_name, query_name);
    if (metric_type == 0) {
        metric = InnerProduct;
    } else {
        metric = L2;
    }

    auto nest_test_vectors = utils::Nest(std::move(queries_vectors), num_query, dim_query);

    base_vectors.resize(num_base * dim_base);
    num_base = base_vectors.size() / dim_base;

    nest_test_vectors.resize(num_query / 500);
    num_query = nest_test_vectors.size();

    dim_gt = 1000;
    num_gt = gt_vectors.size() / dim_gt;

    cout << "Load Data Done!" << endl;

    cout << "Base Vectors: " << num_base << endl;
    cout << "Queries Vectors: " << num_query << endl;
    cout << "Base GT Vectors: " << num_gt << endl;

    cout << "Dimension base_vector: " << dim_base << endl;
    cout << "Dimension query_vector: " << dim_query << endl;
    cout << "Dimension query GT: " << dim_gt << endl;

    size_t k = 1'000;
    size_t num_threads_ = 4;

    utils::Timer query_timer;

    std::vector<std::vector<id_t>> knn(num_query, std::vector<id_t>(k));
    std::vector<std::vector<data_t>> dist(num_query, std::vector<data_t>(k));
    anns::flat::IndexFlat<data_t> index(base_vectors, dim_base, metric);
    index.SetNumThreads(num_threads_);

    query_timer.Reset();
    query_timer.Start();
    index.Search(nest_test_vectors, k, knn, dist);
    query_timer.Stop();
    std::cout << "[Query][FlatCPU] Using GT from file: " << gt_path << std::endl;
    std::cout << "[Query][FlatCPU] Search time: " << query_timer.GetTime() << std::endl;
    std::cout << "[Query][FlatCPU] Recall@" << k << ": " << utils::GetRecall(k, dim_gt, gt_vectors, knn) << std::endl;

    // query_timer.Reset();
    // query_timer.Start();
    // index.Search(nest_train_vectors, k, knn, dist);
    // query_timer.Stop();
    // std::cout << "[Train][FlatCPU] Using GT from file: " << train_gt_path << std::endl;
    // std::cout << "[Train][FlatCPU] Search time: " << query_timer.GetTime() << std::endl;
    // std::cout << "[Train][FlatCPU] Recall@" << k << ": " << utils::GetRecall(k, dtg, train_gt, knn) << std::endl;

    return 0;
}
