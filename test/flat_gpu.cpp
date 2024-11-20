#include <GpuIndexFlat.h>
#include <StandardGpuResources.h>
#include <IndexUtils.h>
#include <DeviceUtils.h>
#include <vector>
#include <numeric>

#include "utils/dataloader.hpp"
#include "../utils/binary_io.hpp"
#include "../utils/resize.hpp"
#include "../utils/timer.hpp"
#include "../utils/recall.hpp"

using data_t = float;
using id_t = uint32_t;
using namespace std;

std::string prefix = "/home/zhengweiguo/liuchengjun/";
std::string idx_prefix = "/data/disk1/liuchengjun/HNNS/checkpoint/";
faiss::MetricType metric;

int main(int argc, char** argv) 
{
    std::vector<data_t> base_vectors, query_vectors, learn_vectors;
    std::vector<id_t> query_gt_vectors, learn_gt_vectors;
    std::string base_name = std::string(argv[1]);
    std::string query_name = std::string(argv[2]);
    size_t k = 1000;
    utils::DataLoader data_loader(base_name, query_name);
    utils::BaseQueryLearnGtConfig cfg;
    std::tie(base_vectors, 
        query_vectors, query_gt_vectors, 
        learn_vectors, learn_gt_vectors,  cfg)
         = data_loader.load_with_learn();
    if (cfg.metric == 0) {
        metric = faiss::MetricType::METRIC_INNER_PRODUCT;
    } else {
        metric = faiss::MetricType::METRIC_L2;
    }
    auto nest_query_vectors = utils::Nest(std::move(query_vectors), cfg.num_query, cfg.dim_query);
    auto nest_learn_vectors = utils::Nest(std::move(learn_vectors), cfg.num_learn, cfg.dim_learn);
    
    cout << "Load Data Done!" << endl;

    size_t ef_construction = 1000;

    utils::Timer query_timer, train_timer;

    int device = 4;
    faiss::gpu::StandardGpuResources res;
    faiss::gpu::GpuIndexFlatConfig config;
    config.device = device;
    faiss::gpu::GpuIndexFlat gpu_index(&res, cfg.dim_base, metric, config);
    gpu_index.add(cfg.num_base, base_vectors.data());

    // query_timer.Reset();    query_timer.Start();
    // gpu_index.search(cfg.num_query, query_vectors.data(), k, dist.data(), knn.data());
    // // gpu_index.search(cfg.num_query, query_vectors_gpu.data(), k, dist.data(), knn.data());
    // query_timer.Stop();
    std::vector<faiss::idx_t> knn(cfg.num_query * k);
    std::vector<data_t> dist(cfg.num_query * k);
    std::vector<std::vector<faiss::idx_t>> nested_knn = 
        utils::Nest(knn, cfg.num_query, k);
    // std::cout << "[Query][GPU] Using GT from file: " << cfg.query_gt_path << std::endl;
    // std::cout << "[Query][GPU] Search time: " << query_timer.GetTime() << std::endl;
    // for (int ck = 1; ck <= k; ck *= 10) {
    //     std::cout << "[Query][GPU] Recall@" << ck << ": " << utils::GetRecall(ck, cfg.dim_query_gt, query_gt_vectors, nested_knn) << std::endl;
    // }

    size_t batch_size = 250000;
    // std::vector<faiss::idx_t> batch_knn(batch_size * k);
    // std::vector<data_t> batch_dist(batch_size * k);
    size_t num_batch = cfg.num_learn / batch_size;
    knn.resize(cfg.num_learn * k);
    dist.resize(cfg.num_learn * k);
    
    query_timer.Reset();    query_timer.Start();
    size_t dim = cfg.dim_learn;
    for (size_t batch = 0; batch < num_batch; batch++) {
        size_t start = batch * batch_size;
        gpu_index.search(batch_size, learn_vectors.data() + start * dim, k, 
            dist.data() + start * k, knn.data() + start * k);
    }
    if (cfg.num_learn % batch_size != 0) {
        size_t start = num_batch * batch_size;
        gpu_index.search(cfg.num_learn - start, learn_vectors.data() + start * dim, k, 
            dist.data() + start * k, knn.data() + start * k);
    }
    query_timer.Stop();
    nested_knn = utils::Nest(knn, cfg.num_learn, k);
    std::cout << "[Train][GPU] Using GT from file: " << cfg.learn_gt_path << std::endl;
    std::cout << "[Train][GPU] Search time: " << query_timer.GetTime() << std::endl;
    for (int ck = 1; ck <= k; ck *= 10) {
        std::cout << "[Train][GPU] Recall@" << ck << ": " << utils::GetRecall(ck, cfg.dim_learn_gt, learn_gt_vectors, nested_knn) << std::endl;
    }
    return 0;
}
// ./flat_gpu wikipedia 96 3000 1024 1000