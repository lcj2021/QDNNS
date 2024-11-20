#include <GpuIndexFlat.h>
#include <StandardGpuResources.h>
#include <IndexUtils.h>
#include <DeviceUtils.h>
#include <numeric>
#include <thread>
#include <vector>
#include <LightGBM/c_api.h>
#include "utils/dataloader.hpp"
#include "graph/hnsw.hpp"

#include "../utils/binary_io.hpp"
#include "../utils/resize.hpp"
#include "../utils/timer.hpp"
#include "../utils/recall.hpp"

using data_t = float;
using id_t = uint32_t;

std::string data_prefix = "/home/zhengweiguo/liuchengjun/anns/";
std::string model_prefix = "/data/disk1/liuchengjun/HNNS/checkpoint/";
std::string label_prefix = "/data/disk1/liuchengjun/HNNS/sample/";
size_t k_gpu = 1000;
float (*metric_cpu)(const data_t *, const data_t *, size_t) = nullptr;
utils::BaseQueryGtConfig cfg;
faiss::MetricType metric_gpu;

std::mt19937 gen(rand());
size_t num_full_feat, dim_full_feat;
size_t k, efq, num_thread; // number of vectors, dimension
std::vector<id_t> train_label;

std::vector<std::vector<id_t>> knn_all;
std::vector<std::vector<data_t>> dist_all;

int main(int argc, char** argv) 
{
    std::vector<data_t> base_vectors, query_vectors, learn_vectors;
    std::vector<id_t> query_gt_vectors, learn_gt_vectors;
    std::string base_name = std::string(argv[1]);
    std::string query_name = std::string(argv[2]);
    size_t M = std::stol(argv[3]);
    efq = std::stol(argv[4]);
    k = std::stol(argv[5]);
    size_t efc = efq;
    num_thread = std::stol(argv[6]);
    size_t check_stamp = std::stol(argv[7]);
    int device = atoi(argv[8]);
    
    utils::DataLoader data_loader(base_name, query_name);
    utils::BaseQueryLearnGtConfig cfg;
    std::tie(base_vectors, 
        query_vectors, query_gt_vectors, 
        learn_vectors, learn_gt_vectors,  cfg)
         = data_loader.load_with_learn();
    if (cfg.metric == 0) {
        metric_cpu = InnerProduct;
        metric_gpu = faiss::MetricType::METRIC_INNER_PRODUCT;
    } else {
        metric_cpu = L2;
        metric_gpu = faiss::MetricType::METRIC_L2;
    }

    auto nest_learn_vectors = utils::Nest(std::move(learn_vectors), cfg.num_learn, cfg.dim_learn);
    auto nest_learn_gt_vectors = utils::Nest(std::move(learn_gt_vectors), cfg.num_learn_gt, cfg.dim_learn_gt);

    std::cout << "Load Data Done!" << std::endl;

    size_t num_check = 100;
    size_t ef_construction = 1000;
    std::string label_path = 
        label_prefix + base_name + "."
        "M_" + std::to_string(M) + "." 
        "efc_" + std::to_string(efc) + "."
        "efs_" + std::to_string(efq) + "."
        "ck_ts_1000.ncheck_" + std::to_string(num_check) + ".recall@1000"
        ".train_label.ivecs";
    auto [num_label, dim_label] = utils::LoadFromFile(train_label, label_path);
    auto nest_label = utils::Nest(std::move(train_label), num_label, dim_label);
    std::cout << "num_label: " << num_label << std::endl;
    std::cout << "dim_label: " << dim_label << std::endl;
    std::vector<std::array<id_t, 2>> number_recall(num_label);
    double avg_recall = 0, avg_NDC = 0;
    for (id_t i = 0; i < num_label; ++i) {
        number_recall[i] = {nest_label[i][1], i};
        avg_recall += number_recall[i][1];
        avg_NDC += number_recall[i][0];
        if (i < 10) {
            std::cout << "label: " << number_recall[i][0] << std::endl;
        }
    }
    avg_recall /= num_label;    avg_NDC /= num_label;
    std::cout << "avg_recall: " << avg_recall << std::endl;
    std::cout << "avg_NDC: " << avg_NDC << std::endl;

    std::sort(number_recall.begin(), number_recall.end(), [&](auto &a, auto &b) {
        return a[0] > b[0];
    });

    std::vector<id_t> sorted_qids(num_label);
    for (id_t i = 0; i < num_label; ++i) {
        sorted_qids[i] = number_recall[i][1];
    }
    std::vector<data_t> sorted_learn(num_label * cfg.dim_learn);
    for (id_t i = 0; i < num_label; ++i) {
        std::copy_n(nest_learn_vectors[sorted_qids[i]].begin(), cfg.dim_learn, sorted_learn.data() + i * cfg.dim_learn);
    }
    auto nest_sorted_learn = utils::Nest(std::move(sorted_learn), num_label, cfg.dim_learn);

    std::string graph_path = 
        "/data/disk1/liuchengjun/HNNS/index/" + base_name + "."
        "M_" + std::to_string(M) + "." 
        "efc_" + std::to_string(ef_construction) + ".hnsw";
    
    std::vector<data_t> test_vector_cpu, test_vector_gpu;
    std::vector<id_t> test_ids_cpu, test_ids_gpu;

    utils::Timer e2e_timer, hnsw_timer, gpu_timer;
    std::cout << "dataset: " << base_name << std::endl;

    std::thread gpu_thread;
    faiss::gpu::StandardGpuResources res;
    faiss::gpu::GpuIndexFlatConfig config;
    config.device = device;
    faiss::gpu::GpuIndexFlat gpu_index(&res, cfg.dim_base, metric_gpu, config);
    gpu_thread = std::thread([&gpu_index, &base_vectors, &cfg] {
        gpu_index.add(cfg.num_base, base_vectors.data());
    });

    auto hnsw = std::make_unique<anns::graph::HNSW<data_t>> (
        base_vectors, graph_path, base_name,
        k, check_stamp, metric_cpu);
    hnsw->SetNumThreads(num_thread);
    gpu_thread.join();

    std::vector<std::vector<id_t>> knn_cpu, knn_all;
    std::vector<std::vector<data_t>> dist_cpu, dist_all;
    hnsw->GetComparisonAndClear();
    size_t nq_cpu, nq_gpu;
    
    // for (int pct = 0; pct <= 100; pct += 5) {
    size_t batch_size = num_label / 10;
    std::vector<faiss::idx_t> knn_gpu(batch_size * k_gpu);
    std::vector<data_t> dist_gpu(batch_size * k_gpu);
    for (size_t batch = 0; batch < num_label; batch += 1) {
        size_t start = batch * batch_size;
        std::vector<std::vector<data_t>> batch_learn_vectors(batch_size, std::vector<data_t>(cfg.dim_learn));
        for (size_t i = 0; i < batch_size; ++i) {
            std::copy_n(nest_sorted_learn[start + i].begin(), cfg.dim_learn, batch_learn_vectors[i].data());
        }
        auto flatten_batch_learn = utils::Flatten(batch_learn_vectors);
        
        double hnsw_time = 0., gpu_time = 0.;
        size_t num_iter = 1;
        // for (int iter = 0; iter < num_iter; ++iter) {
            hnsw_timer.Reset();
            hnsw_timer.Start();
            hnsw->Search(batch_learn_vectors, k, efq, knn_cpu, dist_cpu);
            hnsw_timer.Stop();
            hnsw_time += hnsw_timer.GetTime();

            gpu_timer.Reset();
            gpu_timer.Start();
            gpu_index.search(batch_size, flatten_batch_learn.data(), k_gpu, dist_gpu.data(), knn_gpu.data());
            gpu_timer.Stop();
            gpu_time += gpu_timer.GetTime();
        // }
        std::cout << "hnsw_time: " << hnsw_time / num_iter << std::endl;
        std::cout << "gpu_time: " << gpu_time / num_iter << std::endl;

        auto nested_knn_gpu = utils::Nest(knn_gpu, batch_size, k_gpu);
        std::vector<id_t> batch_gt_vectors(batch_size * cfg.dim_learn_gt);
        for (size_t i = 0; i < batch_size; ++i) {
            std::copy_n(nest_learn_gt_vectors[sorted_qids[start + i]].begin(), cfg.dim_learn_gt, batch_gt_vectors.data() + i * cfg.dim_learn_gt);
        }

        // std::cout << "[Query][GPU] Using GT from file: " << test_gt_path << std::endl;
        std::cout << "[Query][HNSW] avg comparison: " << hnsw->GetComparisonAndClear() / (double)(batch_size) << std::endl;
        for (int ck = k; ck <= k; ck *= 10) {
            size_t num_recall = 0, num_recall_cpu = 0, num_recall_gpu = 0;
            for (int i = 0; i < batch_size; ++i) {
                num_recall_cpu += utils::GetRecallCount(ck, cfg.dim_learn_gt, batch_gt_vectors, knn_cpu[i], i);
            }
            for (int i = 0; i < batch_size; ++i) {
                num_recall_gpu += utils::GetRecallCount(ck, cfg.dim_learn_gt, batch_gt_vectors, nested_knn_gpu[i], i);
            }
            // std::cout << "[Query][CPU] Recall@" << ck << ": " << (double)num_recall_cpu / batch_size / ck << std::endl;
            // std::cout << "[Query][GPU] Recall@" << ck << ": " << (double)num_recall_gpu / batch_size / ck << std::endl;
            std::cout << "[Query][ANNS] Recall@" << ck << ": " << (double)num_recall_cpu / batch_size / ck << std::endl;
            std::cout << "[Query][GPU] Recall@" << ck << ": " << (double)num_recall_gpu / batch_size / ck << std::endl;
        }
    }
    return 0;
}
// ./difficulty wikipedia.base wikipedia.query 32 1000 1000 48 1000