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
    std::vector<data_t> base_vectors, query_vectors, learn_vectors;
    std::vector<id_t> query_gt_vectors, learn_gt_vectors;
    // std::string dataset = "gist1m";
    // std::string dataset = "imagenet";
    // std::string dataset = "wikipedia";
    // std::string dataset = "datacomp-image";
    // std::string dataset = "deep100m";
    std::string base_name = std::string(argv[1]);
    std::string query_name = std::string(argv[2]);
    utils::DataLoader data_loader(base_name, query_name);
    utils::BaseQueryLearnGtConfig cfg;
    std::tie(base_vectors, 
        query_vectors, query_gt_vectors, 
        learn_vectors, learn_gt_vectors,  cfg)
         = data_loader.load_with_learn();
    if (cfg.metric == 0) {
        metric = InnerProduct;
    } else {
        metric = L2;
    }
    auto nest_query_vectors = utils::Nest(std::move(query_vectors), cfg.num_query, cfg.dim_query);
    auto nest_learn_vectors = utils::Nest(std::move(learn_vectors), cfg.num_learn, cfg.dim_learn);

    // // query mask
    // cfg.query_gt_path += ".mask1";
    // for (auto &v : nest_query_vectors) {
    //     for (size_t d = 0; d < cfg.dim_query / 2; ++d) {
    //         v[d] = 0.;
    //     }
    // }

    // base_vectors.resize(cfg.num_base * cfg.dim_base);
    // cfg.num_base = base_vectors.size() / cfg.dim_base;

    // nest_query_vectors.resize(cfg.num_query / 1);
    // cfg.num_query = nest_query_vectors.size();

    cout << "Load Data Done!" << endl;

    cout << "Base Vectors: " << cfg.num_base << endl;
    cout << "Queries Vectors: " << cfg.num_query << endl;

    cout << "Dimension base_vector: " << cfg.dim_base << endl;
    cout << "Dimension query_vector: " << cfg.dim_query << endl;
    std::cout << "Will write to query_gt file: " << cfg.query_gt_path << std::endl;
    std::cout << "Will write to learn_gt file: " << cfg.learn_gt_path << std::endl;


    size_t k = 1'000;
    size_t num_threads_ = 96;

    utils::Timer query_timer;

    std::vector<std::vector<id_t>> knn(cfg.num_query, std::vector<id_t>(k));
    std::vector<std::vector<data_t>> dist(cfg.num_query, std::vector<data_t>(k));
    anns::flat::IndexFlat<data_t> index(base_vectors, cfg.dim_base, metric);
    index.SetNumThreads(num_threads_);

    query_timer.Reset();
    query_timer.Start();
    index.Search(nest_query_vectors, k, knn, dist);
    query_timer.Stop();
    utils::WriteToFile<id_t>(utils::Flatten(knn), {knn.size() * k, 1}, cfg.query_gt_path);
    std::cout << "[Query][FlatCPU] Writing GT to file: " << cfg.query_gt_path << std::endl;
    std::cout << "[Query][FlatCPU] Search time: " << query_timer.GetTime() << std::endl;
    
    query_timer.Reset();
    query_timer.Start();
    index.Search(nest_learn_vectors, k, knn, dist);
    query_timer.Stop();
    utils::WriteToFile<id_t>(utils::Flatten(knn), {knn.size() * k, 1}, cfg.learn_gt_path);
    std::cout << "[Train][FlatCPU] Writing GT to file: " << cfg.learn_gt_path << std::endl;
    std::cout << "[Train][FlatCPU] Search time: " << query_timer.GetTime() << std::endl;


    return 0;
}
