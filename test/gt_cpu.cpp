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
    utils::DataLoader data_loader(base_name, query_name);
    utils::BaseQueryGtConfig cfg;
    std::tie(base_vectors, queries_vectors, gt_vectors, cfg) 
        = data_loader.load();
    if (cfg.metric == 0) {
        metric = InnerProduct;
    } else {
        metric = L2;
    }
    auto nest_test_vectors = utils::Nest(std::move(queries_vectors), cfg.num_query, cfg.dim_query);

    base_vectors.resize(cfg.num_base * cfg.dim_base);
    cfg.num_base = base_vectors.size() / cfg.dim_base;

    nest_test_vectors.resize(cfg.num_query / 1);
    cfg.num_query = nest_test_vectors.size();

    cout << "Load Data Done!" << endl;

    cout << "Base Vectors: " << cfg.num_base << endl;
    cout << "Queries Vectors: " << cfg.num_query << endl;

    cout << "Dimension base_vector: " << cfg.dim_base << endl;
    cout << "Dimension query_vector: " << cfg.dim_query << endl;
    std::cout << "Will write to gt file: " << cfg.query_gt_path << std::endl;


    size_t k = 1'000;
    size_t num_threads_ = 96;

    utils::Timer query_timer;

    std::vector<std::vector<id_t>> knn(cfg.num_query, std::vector<id_t>(k));
    std::vector<std::vector<data_t>> dist(cfg.num_query, std::vector<data_t>(k));
    anns::flat::IndexFlat<data_t> index(base_vectors, cfg.dim_base, metric);
    index.SetNumThreads(num_threads_);

    // query_timer.Reset();
    // query_timer.Start();
    // index.Search(nest_train_vectors, k, knn, dist);
    // query_timer.Stop();
    // utils::WriteToFile<id_t>(utils::Flatten(knn), {knn.size() * k, 1}, train_gt_path);
    // std::cout << "[Train][FlatCPU] Writing GT to file: " << train_gt_path << std::endl;
    // std::cout << "[Train][FlatCPU] Search time: " << query_timer.GetTime() << std::endl;

    query_timer.Reset();
    query_timer.Start();
    index.Search(nest_test_vectors, k, knn, dist);
    query_timer.Stop();
    utils::WriteToFile<id_t>(utils::Flatten(knn), {knn.size() * k, 1}, cfg.query_gt_path);
    std::cout << "[Query][FlatCPU] Writing GT to file: " << cfg.query_gt_path << std::endl;
    std::cout << "[Query][FlatCPU] Search time: " << query_timer.GetTime() << std::endl;

    return 0;
}
