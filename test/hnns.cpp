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
std::string feat_prefix = "/data/disk1/liuchengjun/HNNS/sample/";
size_t k_gpu = 1000;
float (*metric_cpu)(const data_t *, const data_t *, size_t) = nullptr;
utils::BaseQueryGtConfig cfg;
faiss::MetricType metric_gpu;

std::mt19937 gen(rand());
size_t num_full_feat, dim_full_feat;
size_t k, efq, num_thread; // number of vectors, dimension
std::vector<data_t> test_full_feats, train_full_feats;

std::tuple<size_t, size_t> 
partition_random(const std::vector<data_t>& queries_vectors, std::vector<data_t>& part1, std::vector<data_t>& part2, 
        std::vector<id_t>& ids1, std::vector<id_t>& ids2, 
        int dim, int percentage = 50) {
    utils::Timer partition_timer;
    partition_timer.Start();
    assert (0 <= percentage && percentage <= 100 && queries_vectors.size() % dim == 0);
    size_t n = queries_vectors.size() / dim;
    ids1.clear();   part1.clear();
    ids2.clear();   part2.clear();
    if (percentage == 0) {
        part2 = queries_vectors;
        ids2.resize(n);
        std::iota(ids2.begin(), ids2.end(), (id_t)0);
    } else if (percentage == 100) {
        part1 = queries_vectors;
        ids1.resize(n);
        std::iota(ids1.begin(), ids1.end(), (id_t)0);
    } else {
        for (size_t i = 0; i < n; ++i) {
            if (gen() % 100 < percentage) {
                ids1.emplace_back(i);
                part1.insert(part1.end(), queries_vectors.begin() + i * dim, queries_vectors.begin() + (i + 1) * dim);
            } else {
                ids2.emplace_back(i);
                part2.insert(part2.end(), queries_vectors.begin() + i * dim, queries_vectors.begin() + (i + 1) * dim);
            }
        }
    }
    partition_timer.Stop();
    std::cout << "[Partition][Random] Partition time: " << partition_timer.GetTime() << std::endl;
    std::cout << "[Partition][Random] Part1: " << part1.size() / dim << ", part2: " << part2.size() / dim << std::endl;
    return std::make_tuple(part1.size() / dim, part2.size() / dim);
}

std::vector<std::vector<id_t>> knn_all;
std::vector<std::vector<data_t>> dist_all;
std::vector<id_t> qids_all;
BoosterHandle handle_classfication, handle_regression; // LightGBM model

std::tuple<size_t, size_t> 
partition_hnns(const std::vector<data_t>& queries_vectors, std::vector<data_t>& part1, std::vector<data_t>& part2, 
        std::vector<id_t>& ids1, std::vector<id_t>& ids2, 
        int dim, anns::graph::HNSW<data_t>& hnsw, int percentage = 50) {
    utils::Timer partition_timer;
    partition_timer.Start();
    assert (0 <= percentage && percentage <= 100 && queries_vectors.size() % dim == 0);
    size_t n = queries_vectors.size() / dim;
    ids1.clear();   part1.clear();
    ids2.clear();   part2.clear();
    // if (percentage == 0) {
    //     part2 = queries_vectors;
    //     ids2.resize(n);
    //     std::iota(ids2.begin(), ids2.end(), (id_t)0);
    // } else if (percentage == 100) {
    //     part1 = queries_vectors;
    //     ids1.resize(n);
    //     std::iota(ids1.begin(), ids1.end(), (id_t)0);
    // } else {
        hnsw.SearchHNNS(utils::Nest(std::move(queries_vectors), queries_vectors.size() / cfg.dim_base, cfg.dim_base), 
            k, efq, knn_all, dist_all, qids_all, 0);
        // size_t NDC_avg = 0, NDC_max = 0, NDC_min = 1e9;
        // for (int i = 0; i < n; ++i) {
        //     NDC_avg += hnsw.test_inter_results[i].NDC;
        //     NDC_max = std::max(NDC_max, hnsw.test_inter_results[i].NDC);
        //     NDC_min = std::min(NDC_min, hnsw.test_inter_results[i].NDC);
        // }
        // std::cout << (double)NDC_avg / n << std::endl;
        // std::cout << NDC_min << " " << NDC_max << std::endl;
        // hnsw.GetComparisonAndClear();
        auto scores = utils::Flatten(dist_all);
        
        float threshold;
        auto scores_backup = scores;
        size_t idx = std::min(n * percentage / 100, scores.size() - 1);
        nth_element(scores_backup.begin(), scores_backup.begin() + idx, scores_backup.end());
        threshold = scores_backup[idx];
        std::cout << "[Partition][HNNS] Threshold: " << threshold << std::endl;

        for (size_t i = 0; i < n; ++i) {
            if (scores[i] < threshold) {
                ids1.emplace_back(i);
                part1.insert(part1.end(), queries_vectors.begin() + i * dim, queries_vectors.begin() + (i + 1) * dim);
            } else {
                ids2.emplace_back(i);
                part2.insert(part2.end(), queries_vectors.begin() + i * dim, queries_vectors.begin() + (i + 1) * dim);
            }
        }
    // }
    partition_timer.Stop();
    std::cout << "[Partition][HNNS] Partition time: " << partition_timer.GetTime() << std::endl;
    std::cout << "[Partition][HNNS] Part1: " << part1.size() / dim << ", part2: " << part2.size() / dim << std::endl;
    return std::make_tuple(part1.size() / dim, part2.size() / dim);
}

std::tuple<size_t, size_t> 
partition_hnns_qonly(const std::vector<data_t>& queries_vectors, std::vector<data_t>& part1, std::vector<data_t>& part2, 
        std::vector<id_t>& ids1, std::vector<id_t>& ids2, 
        int dim, anns::graph::HNSW<data_t>& hnsw, int percentage = 50) {
    utils::Timer partition_timer;
    partition_timer.Start();
    assert (0 <= percentage && percentage <= 100 && queries_vectors.size() % dim == 0);
    size_t n = queries_vectors.size() / dim;
    ids1.clear();   part1.clear();
    ids2.clear();   part2.clear();
    // if (percentage == 0) {
    //     part2 = queries_vectors;
    //     ids2.resize(n);
    //     std::iota(ids2.begin(), ids2.end(), (id_t)0);
    // } else if (percentage == 100) {
    //     part1 = queries_vectors;
    //     ids1.resize(n);
    //     std::iota(ids1.begin(), ids1.end(), (id_t)0);
    // } else {
        int64_t out_len;
        double out_result;
        std::vector<double> scores(n);
        std::string params = "num_threads=" + std::to_string(num_thread);
        LGBM_BoosterPredictForMat(handle_classfication, queries_vectors.data(), C_API_DTYPE_FLOAT32, 
            n, dim, 1, C_API_PREDICT_NORMAL, 0, -1, params.data(), &out_len, scores.data());
        
        float threshold;
        auto scores_backup = scores;
        size_t idx = std::min(n * percentage / 100, scores.size() - 1);
        nth_element(scores_backup.begin(), scores_backup.begin() + idx, scores_backup.end());
        threshold = scores_backup[idx];
        std::cout << "[Partition][HNNS] Threshold: " << threshold << std::endl;

        for (size_t i = 0; i < n; ++i) {
            if (scores[i] < threshold) {
                ids1.emplace_back(i);
                part1.insert(part1.end(), queries_vectors.begin() + i * dim, queries_vectors.begin() + (i + 1) * dim);
            } else {
                ids2.emplace_back(i);
                part2.insert(part2.end(), queries_vectors.begin() + i * dim, queries_vectors.begin() + (i + 1) * dim);
            }
        }
    // }
    partition_timer.Stop();
    std::cout << "[Partition][HNNS] Partition time: " << partition_timer.GetTime() << std::endl;
    std::cout << "[Partition][HNNS] Part1: " << part1.size() / dim << ", part2: " << part2.size() / dim << std::endl;
    return std::make_tuple(part1.size() / dim, part2.size() / dim);
}

std::tuple<size_t, size_t> 
partition_hnns_full_feat(const std::vector<data_t>& queries_vectors, std::vector<data_t>& part1, std::vector<data_t>& part2, 
        std::vector<id_t>& ids1, std::vector<id_t>& ids2, 
        int dim, const std::vector<data_t>& test_full_feat, int percentage = 50) {
    utils::Timer partition_timer;
    partition_timer.Start();
    assert (0 <= percentage && percentage <= 100 && queries_vectors.size() % dim == 0);
    size_t n = queries_vectors.size() / dim;
    std::cout << "[Partition][HNNS] dim_feat: " << dim_full_feat << std::endl;
    ids1.clear();   part1.clear();
    ids2.clear();   part2.clear();
    if (percentage == 0) {
        part2 = queries_vectors;
        ids2.resize(n);
        std::iota(ids2.begin(), ids2.end(), (id_t)0);
    } else if (percentage == 100) {
        part1 = queries_vectors;
        ids1.resize(n);
        std::iota(ids1.begin(), ids1.end(), (id_t)0);
    } else {
        int64_t out_len;
        double out_result;
        std::vector<double> scores(n);
        std::string params = "num_threads=" + std::to_string(num_thread);
        LGBM_BoosterPredictForMat(handle_classfication, test_full_feat.data(), C_API_DTYPE_FLOAT32, 
            n, dim_full_feat, 1, C_API_PREDICT_NORMAL, 0, -1, params.data(), &out_len, scores.data());
        
        float threshold;
        auto scores_backup = scores;
        size_t idx = std::min(n * percentage / 100, scores.size() - 1);
        nth_element(scores_backup.begin(), scores_backup.begin() + idx, scores_backup.end());
        threshold = scores_backup[idx];
        std::cout << "[Partition][HNNS] Threshold: " << threshold << std::endl;

        for (size_t i = 0; i < n; ++i) {
            if (scores[i] < threshold) {
                ids1.emplace_back(i);
                part1.insert(part1.end(), queries_vectors.begin() + i * dim, queries_vectors.begin() + (i + 1) * dim);
            } else {
                ids2.emplace_back(i);
                part2.insert(part2.end(), queries_vectors.begin() + i * dim, queries_vectors.begin() + (i + 1) * dim);
            }
        }
    }
    partition_timer.Stop();
    std::cout << "[Partition][HNNS] Partition time: " << partition_timer.GetTime() << std::endl;
    std::cout << "[Partition][HNNS] Part1: " << part1.size() / dim << ", part2: " << part2.size() / dim << std::endl;
    return std::make_tuple(part1.size() / dim, part2.size() / dim);
}

std::tuple<size_t, size_t> 
partition_hnns_approximate_feat(const std::vector<data_t>& queries_vectors, std::vector<data_t>& part1, std::vector<data_t>& part2, 
        std::vector<id_t>& ids1, std::vector<id_t>& ids2, 
        int dim, const std::vector<data_t>& train_full_feat, anns::graph::HNSW<data_t>& hnsw, int percentage = 50) {
    utils::Timer partition_timer;
    partition_timer.Start();
    assert (0 <= percentage && percentage <= 100 && queries_vectors.size() % dim == 0);
    size_t n = cfg.num_query, num_train = train_full_feat.size() / dim_full_feat;
    std::cout << "[Partition][HNNS] dim_feat: " << dim_full_feat << std::endl;
    std::cout << "[Partition][HNNS] num_train: " << train_full_feat.size() / dim_full_feat << std::endl;
    ids1.clear();   part1.clear();
    ids2.clear();   part2.clear();
    if (percentage == 0) {
        part2 = queries_vectors;
        ids2.resize(cfg.num_query);
        std::iota(ids2.begin(), ids2.end(), (id_t)0);
    } else if (percentage == 100) {
        part1 = queries_vectors;
        ids1.resize(cfg.num_query);
        std::iota(ids1.begin(), ids1.end(), (id_t)0);
    } else {
        
        auto nest_test_vectors = utils::Nest(std::move(queries_vectors), cfg.num_query, cfg.dim_query);
        hnsw.Search(nest_test_vectors, 1, 50, knn_all, dist_all);
        std::vector<data_t> feat_approximate(
            cfg.num_query * dim_full_feat
        );

    #pragma omp parallel for num_threads(num_thread)
        for (size_t qid = 0; qid < cfg.num_query; ++qid) {
            auto fv = feat_approximate.data() + qid * dim_full_feat;
            size_t similar_id = knn_all[qid][0];
            assert (similar_id < num_train);
            for (int i = 0; i < cfg.dim_query; ++i) {
                fv[i] = queries_vectors[qid * cfg.dim_query + i];
            }
            for (int i = cfg.dim_query; i < dim_full_feat; ++i) {
                fv[i] = train_full_feats[similar_id * dim_full_feat + cfg.dim_query + i];
            }
        }

        int64_t out_len;
        double out_result;
        std::vector<double> scores(n);
        std::string params = "num_threads=" + std::to_string(num_thread);
        LGBM_BoosterPredictForMat(handle_classfication, feat_approximate.data(), C_API_DTYPE_FLOAT32, 
            n, dim_full_feat, 1, C_API_PREDICT_NORMAL, 0, -1, params.data(), &out_len, scores.data());

        float threshold;
        auto scores_backup = scores;
        size_t idx = std::min(n * percentage / 100, scores.size() - 1);
        nth_element(scores_backup.begin(), scores_backup.begin() + idx, scores_backup.end());
        threshold = scores_backup[idx];
        std::cout << "[Partition][HNNS] Threshold: " << threshold << std::endl;

        for (size_t i = 0; i < n; ++i) {
            if (scores[i] < threshold) {
                ids1.emplace_back(i);
                part1.insert(part1.end(), queries_vectors.begin() + i * dim, queries_vectors.begin() + (i + 1) * dim);
            } else {
                ids2.emplace_back(i);
                part2.insert(part2.end(), queries_vectors.begin() + i * dim, queries_vectors.begin() + (i + 1) * dim);
            }
        }
    }
    partition_timer.Stop();
    std::cout << "[Partition][HNNS] Partition time: " << partition_timer.GetTime() << std::endl;
    std::cout << "[Partition][HNNS] Part1: " << part1.size() / dim << ", part2: " << part2.size() / dim << std::endl;
    return std::make_tuple(part1.size() / dim, part2.size() / dim);
}

std::tuple<size_t, size_t> 
partition_hnns_MTL(const std::vector<data_t>& queries_vectors, std::vector<data_t>& part1, std::vector<data_t>& part2, 
        std::vector<id_t>& ids1, std::vector<id_t>& ids2, 
        int dim, int percentage = 50) {
    utils::Timer partition_timer;
    partition_timer.Start();
    assert (0 <= percentage && percentage <= 100 && queries_vectors.size() % dim == 0);
    size_t n = queries_vectors.size() / dim;
    ids1.clear();   part1.clear();
    ids2.clear();   part2.clear();
    if (percentage == 0) {
        part2 = queries_vectors;
        ids2.resize(n);
        std::iota(ids2.begin(), ids2.end(), (id_t)0);
    } else if (percentage == 100) {
        part1 = queries_vectors;
        ids1.resize(n);
        std::iota(ids1.begin(), ids1.end(), (id_t)0);
    } else {
        int64_t out_len;
        double out_result;
        std::vector<double> scores_classification(n), scores_regression(n), scores_combined(n);
        std::string params = "num_threads=" + std::to_string(num_thread);
        LGBM_BoosterPredictForMat(handle_classfication, queries_vectors.data(), C_API_DTYPE_FLOAT32, 
            n, dim, 1, C_API_PREDICT_NORMAL, 0, -1, params.data(), &out_len, scores_classification.data());
        LGBM_BoosterPredictForMat(handle_regression, queries_vectors.data(), C_API_DTYPE_FLOAT32, 
            n, dim, 1, C_API_PREDICT_NORMAL, 0, -1, params.data(), &out_len, scores_regression.data());

        for (size_t i = 0; i < n; ++i) {
            scores_combined[i] = scores_classification[i] * scores_regression[i];
        }

        float threshold;
        auto scores_backup = scores_combined;
        size_t idx = std::min(n * percentage / 100, scores_combined.size() - 1);
        nth_element(scores_backup.begin(), scores_backup.begin() + idx, scores_backup.end());
        threshold = scores_backup[idx];
        std::cout << "[Partition][HNNS] Threshold: " << threshold << std::endl;

        for (size_t i = 0; i < n; ++i) {
            if (scores_combined[i] < threshold) {
                ids1.emplace_back(i);
                part1.insert(part1.end(), queries_vectors.begin() + i * dim, queries_vectors.begin() + (i + 1) * dim);
            } else {
                ids2.emplace_back(i);
                part2.insert(part2.end(), queries_vectors.begin() + i * dim, queries_vectors.begin() + (i + 1) * dim);
            }
        }
    }
    partition_timer.Stop();
    std::cout << "[Partition][HNNS] Partition time: " << partition_timer.GetTime() << std::endl;
    std::cout << "[Partition][HNNS] Part1: " << part1.size() / dim << ", part2: " << part2.size() / dim << std::endl;
    return std::make_tuple(part1.size() / dim, part2.size() / dim);
}

int main(int argc, char** argv) 
{
    std::vector<data_t> base_vectors, queries_vectors, learn_vectors;
    // std::vector<id_t> gt_vectors, train_gt;
    std::vector<id_t> gt_vectors;
    // std::string dataset = "gist1m";
    // std::string dataset = "imagenet";
    // std::string dataset = "wikipedia";
    std::string base_name = std::string(argv[1]);
    std::string query_name = std::string(argv[2]);
    size_t M = std::stol(argv[3]);
    efq = std::stol(argv[4]);
    k = std::stol(argv[5]);
    size_t efc = efq;
    num_thread = std::stol(argv[6]);
    size_t check_stamp = std::stol(argv[7]);
    size_t threshold = std::stol(argv[8]);
    std::string method = std::string(argv[9]);
    size_t num_cross = 0;
    
    utils::DataLoader data_loader(base_name, query_name);
    std::tie(base_vectors, queries_vectors, gt_vectors, cfg)
         = data_loader.load();
    if (cfg.metric == 0) {
        metric_cpu = InnerProduct;
        metric_gpu = faiss::MetricType::METRIC_INNER_PRODUCT;
    } else {
        metric_cpu = L2;
        metric_gpu = faiss::MetricType::METRIC_L2;
    }

    auto nest_test_vectors = utils::Nest(std::move(queries_vectors), cfg.num_query, cfg.dim_query);

    nest_test_vectors.resize(cfg.num_query);
    cfg.num_query = nest_test_vectors.size();

    std::cout << "Load Data Done!" << std::endl;

    // std::string test_full_feats_path = feat_prefix + base_name + 
    //     ".M_" + std::to_string(M) + ".efc_" + std::to_string(efc) + 
    //     ".efs_" + std::to_string(efc) + 
    //     ".ck_ts_" + std::to_string(check_stamp) + 
    //     ".ncheck_100.recall@1000" + ".test_feats_nn.fvecs";
    // std::string train_full_feats_path = feat_prefix + base_name + 
    //     ".M_" + std::to_string(M) + ".efc_" + std::to_string(efc) + 
    //     ".efs_" + std::to_string(efc) + 
    //     ".ck_ts_" + std::to_string(check_stamp) + 
    //     ".ncheck_100.recall@1000" + ".train_feats_nn.fvecs";

    // std::cout << "test_full_feats_path: " << test_full_feats_path << std::endl;
    // utils::LoadFromFile(test_full_feats, test_full_feats_path);
    // std::cout << "train_full_feats_path: " << train_full_feats_path << std::endl;
    // std::tie(num_full_feat, dim_full_feat) = utils::LoadFromFile(train_full_feats, train_full_feats_path);

    size_t num_check = 100;
    size_t ef_construction = 1000;
    std::string graph_path = 
        "/data/disk1/liuchengjun/HNNS/index/" + base_name + "."
        "M_" + std::to_string(M) + "." 
        "efc_" + std::to_string(ef_construction) + ".hnsw";
    
    std::string model_classification_path = "/data/disk1/liuchengjun/HNNS/checkpoint/" + base_name + 
        ".M_" + std::to_string(M) + ".efc_" + std::to_string(efc) + 
        ".efs_" + std::to_string(efc) + 
        ".ck_ts_" + std::to_string(check_stamp) + 
        ".ncheck_100.recall@1000.thr_" + std::to_string(threshold) + 
        ".classification.cross_" + std::to_string(num_cross);
    if (method == "hnns_qonly" || method == "hnns_MTL") model_classification_path += ".qonly";
    model_classification_path += ".txt";

    // std::string model_regression_path = "/data/disk1/liuchengjun/HNNS/checkpoint/" + base_name + 
    //     ".M_" + std::to_string(M) + ".efc_" + std::to_string(efc) + 
    //     ".efs_" + std::to_string(efc) + 
    //     ".ck_ts_" + std::to_string(check_stamp) + 
    //     ".ncheck_100.recall@1000.thr_" + std::to_string(threshold) + 
    //     ".regression.cross_" + std::to_string(num_cross);
    // if (method == "hnns_qonly" || method == "hnns_MTL") model_regression_path += ".qonly";
    // model_regression_path += ".txt";

    std::vector<data_t> test_vector_cpu, test_vector_gpu;
    std::vector<id_t> test_ids_cpu, test_ids_gpu;

    utils::Timer e2e_timer, hnsw_timer;
    std::cout << "dataset: " << base_name << std::endl;

    std::vector<faiss::idx_t> knn_gpu(cfg.num_query * k_gpu);
    std::vector<data_t> dist_gpu(cfg.num_query * k_gpu);

    std::thread gpu_thread;
    int device = 7;
    faiss::gpu::StandardGpuResources res;
    faiss::gpu::GpuIndexFlatConfig config;
    config.device = device;
    faiss::gpu::GpuIndexFlat gpu_index(&res, cfg.dim_base, metric_gpu, config);
    gpu_thread = std::thread([&gpu_index, &base_vectors] {
        gpu_index.add(cfg.num_base, base_vectors.data());
    });

    std::string query_dataset = utils::split(query_name, '.')[0];
    std::string graph_learn_path = 
        "/data/disk1/liuchengjun/HNNS/index/" + query_dataset + ".learn."
        "M_" + std::to_string(8) + "." 
        "efc_" + std::to_string(50) + ".hnsw";

    auto hnsw = std::make_unique<anns::graph::HNSW<data_t>> (
        base_vectors, graph_path, base_name,
        k, check_stamp, metric_cpu);
    // // replace the last "query" with "learn"
    // size_t last_query_pos = cfg.query_path.find("query.");
    // assert (last_query_pos != std::string::npos);
    // std::string learn_path = cfg.query_path.substr(0, last_query_pos) + "learn." + cfg.query_path.substr(last_query_pos + 6);
    // auto [num_train, dim_train] = utils::LoadFromFile(learn_vectors, learn_path);
    // std::cout << "num_train: " << num_train << ", dim_train: " << dim_train << std::endl;
    // auto hnsw_learn = std::make_unique<anns::graph::HNSW<data_t>> (
    //     learn_vectors, graph_learn_path, query_dataset + ".learn",
    //     k, check_stamp, metric_cpu);
    hnsw->SetNumThreads(num_thread);
    gpu_thread.join();

    std::cout << "[LightGBM] model_classification_path: " << model_classification_path << std::endl;
    int out_num_iterations, result;
    result = LGBM_BoosterCreateFromModelfile(model_classification_path.data(), &out_num_iterations, &handle_classfication);
    if (result == 0) {
        hnsw->LoadLightGBM(handle_classfication);
        std::cout << "[LightGBM] Model loaded successfully." << std::endl;
    } else {
        std::cout << "[LightGBM] Failed to load model." << std::endl;
    }
    // std::cout << "[LightGBM] model_classification_path: " << model_regression_path << std::endl;
    // result = LGBM_BoosterCreateFromModelfile(model_regression_path.data(), &out_num_iterations, &handle_regression);
    if (result == 0) {
        // hnsw->LoadLightGBM(handle_regression);
        std::cout << "[LightGBM] Model loaded successfully." << std::endl;
    } else {
        std::cout << "[LightGBM] Failed to load model." << std::endl;
    }

    std::vector<std::vector<id_t>> knn_cpu, knn_all;
    std::vector<std::vector<data_t>> dist_cpu, dist_all;
    qids_all.resize(cfg.num_query);
    std::iota(qids_all.begin(), qids_all.end(), (id_t)0);
    hnsw->GetComparisonAndClear();
    size_t nq_cpu, nq_gpu;
    std::vector<int> pcts = {
        // 0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50
        // , 55, 60, 65, 
        // 70, 72, 74, 76, 78, 
        // 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 
        // 95, 96, 97, 98, 99, 100
    };
    for (int pct = 0; pct <= 100; pct += 2) {
        pcts.push_back(pct);
    }
    std::reverse(pcts.begin(), pcts.end());
    
    // for (int pct = 0; pct <= 100; pct += 5) {
    for (auto pct : pcts) {
        for (int iter = 0; iter < 3; ++iter) {

            // Random
            if (method == "random") {
                e2e_timer.Reset();    e2e_timer.Start();
                std::tie(nq_cpu, nq_gpu) = partition_random(queries_vectors, test_vector_cpu, test_vector_gpu, test_ids_cpu, test_ids_gpu, cfg.dim_base, pct);
                knn_gpu.resize(nq_gpu * k_gpu);
                dist_gpu.resize(nq_gpu * k_gpu);

                if (nq_gpu > 0) {
                    gpu_thread = std::thread([&gpu_index, &test_vector_gpu, &knn_gpu, &dist_gpu, &nq_gpu] {
                        gpu_index.search(nq_gpu, test_vector_gpu.data(), k_gpu, dist_gpu.data(), knn_gpu.data());
                    });
                }
                hnsw_timer.Reset();    hnsw_timer.Start();
                hnsw->Search(utils::Nest(std::move(test_vector_cpu), test_vector_cpu.size() / cfg.dim_query, cfg.dim_query), 
                    k, efq, knn_cpu, dist_cpu);
                hnsw_timer.Stop();
                if (nq_gpu > 0) {
                    gpu_thread.join();
                }
                e2e_timer.Stop();
                std::cout << "[Query][Random] HNSW time: " << hnsw_timer.GetTime() << std::endl;
                std::cout << "[Query][Random] E2E time: " << e2e_timer.GetTime() << std::endl;
            } else if (method == "hnns") {
                // HNNS
                knn_all.clear();         dist_all.clear();
                e2e_timer.Reset();    e2e_timer.Start();
                std::tie(nq_cpu, nq_gpu) = partition_hnns(queries_vectors, test_vector_cpu, test_vector_gpu, test_ids_cpu, test_ids_gpu, cfg.dim_base, *hnsw, pct);
                knn_gpu.resize(nq_gpu * k_gpu);
                dist_gpu.resize(nq_gpu * k_gpu);

                if (nq_gpu > 0) {
                    gpu_thread = std::thread([&gpu_index, &test_vector_gpu, &knn_gpu, &dist_gpu, &nq_gpu] {
                        gpu_index.search(nq_gpu, test_vector_gpu.data(), k_gpu, dist_gpu.data(), knn_gpu.data());
                    });
                }
                hnsw_timer.Reset();    hnsw_timer.Start();
                hnsw->SearchHNNS(utils::Nest(std::move(test_vector_cpu), test_vector_cpu.size() / cfg.dim_query, cfg.dim_query), 
                    k, efq, knn_cpu, dist_cpu, test_ids_cpu, 1);
                // hnsw->Search(utils::Nest(std::move(test_vector_cpu), test_vector_cpu.size() / cfg.dim_query, cfg.dim_query), 
                //     k, efq, knn_cpu, dist_cpu);
                hnsw_timer.Stop();
                if (nq_gpu > 0) {
                    gpu_thread.join();
                }
                e2e_timer.Stop();
                // hnsw->Reset();
                std::cout << "[Query][HNNS] HNSW time: " << hnsw_timer.GetTime() << std::endl;
                std::cout << "[Query][HNNS] E2E time: " << e2e_timer.GetTime() << std::endl;
            } else if (method == "hnns_qonly") {
                knn_all.clear();         dist_all.clear();
                e2e_timer.Reset();    e2e_timer.Start();
                std::tie(nq_cpu, nq_gpu) = partition_hnns_qonly(queries_vectors, test_vector_cpu, test_vector_gpu, test_ids_cpu, test_ids_gpu, cfg.dim_base, *hnsw, pct);
                knn_gpu.resize(nq_gpu * k_gpu);
                dist_gpu.resize(nq_gpu * k_gpu);

                if (nq_gpu > 0) {
                    gpu_thread = std::thread([&gpu_index, &test_vector_gpu, &knn_gpu, &dist_gpu, &nq_gpu] {
                        gpu_index.search(nq_gpu, test_vector_gpu.data(), k_gpu, dist_gpu.data(), knn_gpu.data());
                    });
                }
                hnsw_timer.Reset();    hnsw_timer.Start();
                hnsw->Search(utils::Nest(std::move(test_vector_cpu), test_vector_cpu.size() / cfg.dim_query, cfg.dim_query), 
                    k, efq, knn_cpu, dist_cpu);
                hnsw_timer.Stop();
                if (nq_gpu > 0) {
                    gpu_thread.join();
                }
                e2e_timer.Stop();
                std::cout << "[Query][HNNS] HNSW time: " << hnsw_timer.GetTime() << std::endl;
                std::cout << "[Query][HNNS] E2E time: " << e2e_timer.GetTime() << std::endl;
            } else if (method == "hnns_full") {
                knn_all.clear();         dist_all.clear();
                e2e_timer.Reset();    e2e_timer.Start();
                std::tie(nq_cpu, nq_gpu) = partition_hnns_full_feat(queries_vectors, test_vector_cpu, test_vector_gpu, test_ids_cpu, test_ids_gpu, cfg.dim_base, test_full_feats, pct);
                knn_gpu.resize(nq_gpu * k_gpu);
                dist_gpu.resize(nq_gpu * k_gpu);

                if (nq_gpu > 0) {
                    gpu_thread = std::thread([&gpu_index, &test_vector_gpu, &knn_gpu, &dist_gpu, &nq_gpu] {
                        gpu_index.search(nq_gpu, test_vector_gpu.data(), k_gpu, dist_gpu.data(), knn_gpu.data());
                    });
                }
                hnsw_timer.Reset();    hnsw_timer.Start();
                hnsw->Search(utils::Nest(std::move(test_vector_cpu), test_vector_cpu.size() / cfg.dim_query, cfg.dim_query), 
                    k, efq, knn_cpu, dist_cpu);
                hnsw_timer.Stop();
                if (nq_gpu > 0) {
                    gpu_thread.join();
                }
                e2e_timer.Stop();
                std::cout << "[Query][HNNS] HNSW time: " << hnsw_timer.GetTime() << std::endl;
                std::cout << "[Query][HNNS] E2E time: " << e2e_timer.GetTime() << std::endl;
            } else if (method == "hnns_approximate") {
                // knn_all.clear();         dist_all.clear();
                // e2e_timer.Reset();    e2e_timer.Start();
                // std::tie(nq_cpu, nq_gpu) = partition_hnns_approximate_feat(queries_vectors, test_vector_cpu, test_vector_gpu, test_ids_cpu, test_ids_gpu, cfg.dim_base, train_full_feats, *hnsw_learn, pct);
                // knn_gpu.resize(nq_gpu * k_gpu);
                // dist_gpu.resize(nq_gpu * k_gpu);

                // if (nq_gpu > 0) {
                //     gpu_thread = std::thread([&gpu_index, &test_vector_gpu, &knn_gpu, &dist_gpu, &nq_gpu] {
                //         gpu_index.search(nq_gpu, test_vector_gpu.data(), k_gpu, dist_gpu.data(), knn_gpu.data());
                //     });
                // }
                // hnsw_timer.Reset();    hnsw_timer.Start();
                // hnsw->Search(utils::Nest(std::move(test_vector_cpu), test_vector_cpu.size() / cfg.dim_query, cfg.dim_query), 
                //     k, efq, knn_cpu, dist_cpu);
                // hnsw_timer.Stop();
                // if (nq_gpu > 0) {
                //     gpu_thread.join();
                // }
                // e2e_timer.Stop();
                // std::cout << "[Query][HNNS] HNSW time: " << hnsw_timer.GetTime() << std::endl;
                // std::cout << "[Query][HNNS] E2E time: " << e2e_timer.GetTime() << std::endl;
            } else if (method == "hnns_MTL") {
                knn_all.clear();         dist_all.clear();
                e2e_timer.Reset();    e2e_timer.Start();
                std::tie(nq_cpu, nq_gpu) = partition_hnns_MTL(queries_vectors, test_vector_cpu, test_vector_gpu, test_ids_cpu, test_ids_gpu, cfg.dim_base, pct);
                knn_gpu.resize(nq_gpu * k_gpu);
                dist_gpu.resize(nq_gpu * k_gpu);

                if (nq_gpu > 0) {
                    gpu_thread = std::thread([&gpu_index, &test_vector_gpu, &knn_gpu, &dist_gpu, &nq_gpu] {
                        gpu_index.search(nq_gpu, test_vector_gpu.data(), k_gpu, dist_gpu.data(), knn_gpu.data());
                    });
                }
                hnsw_timer.Reset();    hnsw_timer.Start();
                hnsw->Search(utils::Nest(std::move(test_vector_cpu), test_vector_cpu.size() / cfg.dim_query, cfg.dim_query), 
                    k, efq, knn_cpu, dist_cpu);
                hnsw_timer.Stop();
                if (nq_gpu > 0) {
                    gpu_thread.join();
                }
                e2e_timer.Stop();
                std::cout << "[Query][HNNS] HNSW time: " << hnsw_timer.GetTime() << std::endl;
                std::cout << "[Query][HNNS] E2E time: " << e2e_timer.GetTime() << std::endl;
            } else if (method == "hnns_full") {
                knn_all.clear();         dist_all.clear();
                e2e_timer.Reset();    e2e_timer.Start();
                std::tie(nq_cpu, nq_gpu) = partition_hnns_full_feat(queries_vectors, test_vector_cpu, test_vector_gpu, test_ids_cpu, test_ids_gpu, cfg.dim_query, test_full_feats, pct);
                knn_gpu.resize(nq_gpu * k_gpu);
                dist_gpu.resize(nq_gpu * k_gpu);

                if (nq_gpu > 0) {
                    gpu_thread = std::thread([&gpu_index, &test_vector_gpu, &knn_gpu, &dist_gpu, &nq_gpu] {
                        gpu_index.search(nq_gpu, test_vector_gpu.data(), k_gpu, dist_gpu.data(), knn_gpu.data());
                    });
                }
                hnsw_timer.Reset();    hnsw_timer.Start();
                hnsw->Search(utils::Nest(std::move(test_vector_cpu), test_vector_cpu.size() / cfg.dim_query, cfg.dim_query), 
                    k, efq, knn_cpu, dist_cpu);
                hnsw_timer.Stop();
                if (nq_gpu > 0) {
                    gpu_thread.join();
                }
                e2e_timer.Stop();
                std::cout << "[Query][HNNS] HNSW time: " << hnsw_timer.GetTime() << std::endl;
                std::cout << "[Query][HNNS] E2E time: " << e2e_timer.GetTime() << std::endl;
            }



        }

        auto nested_knn_gpu = utils::Nest(knn_gpu, nq_gpu, k_gpu);
        // std::cout << "[Query][GPU] Using GT from file: " << test_gt_path << std::endl;
        for (int ck = k; ck <= k; ck *= 10) {
            size_t num_recall = 0, num_recall_cpu = 0, num_recall_gpu = 0;
            for (int i = 0; i < test_ids_cpu.size(); ++i) {
                num_recall_cpu += utils::GetRecallCount(ck, cfg.dim_query_gt, gt_vectors, knn_cpu[i], test_ids_cpu[i]);
            }
            for (int i = 0; i < test_ids_gpu.size(); ++i) {
                num_recall_gpu += utils::GetRecallCount(ck, cfg.dim_query_gt, gt_vectors, nested_knn_gpu[i], test_ids_gpu[i]);
            }
            num_recall = num_recall_cpu + num_recall_gpu;
            // std::cout << "[Query][CPU] Recall@" << ck << ": " << (double)num_recall_cpu / test_ids_cpu.size() / ck << std::endl;
            // std::cout << "[Query][GPU] Recall@" << ck << ": " << (double)num_recall_gpu / test_ids_gpu.size() / ck << std::endl;
            std::string pct_str = "CPU" + std::to_string(pct) + "% + ""GPU" + std::to_string(100 - pct) + "%";
            std::cout << "[Query][HNNS = " << pct_str << "] Recall@" << ck << ": " << (double)num_recall / cfg.num_query / ck << std::endl;
            std::cout << "[Query][HNSW] avg comparison: " << hnsw->GetComparisonAndClear() / (double)(test_vector_cpu.size() / cfg.dim_query) << std::endl;
        }
    }
    return 0;
}
// nohup ./hnns datacomp-comp 32 1000 1000 48 1000 hnns > ../log/datacomp-image.M_32.efc_1000.efs_1000.ck_ts_1000.ncheck_100.recall@1000.nthread_48.hnns_optimized.log &