#include <GpuIndexFlat.h>
#include <StandardGpuResources.h>
#include <IndexUtils.h>
#include <DeviceUtils.h>
#include <numeric>
#include <thread>
#include <vector>
#include <LightGBM/c_api.h>
#include "graph/hnsw.hpp"

#include "../utils/binary_io.hpp"
#include "../utils/resize.hpp"
#include "../utils/timer.hpp"
#include "../utils/recall.hpp"

using data_t = float;
using id_t = uint32_t;
using namespace std;

std::string prefix = "/home/zhengweiguo/liuchengjun/";
std::string idx_prefix = "/data/disk1/liuchengjun/HNNS/checkpoint/";
size_t k_gpu = 1000;
float (*metric_cpu)(const data_t *, const data_t *, size_t) = nullptr;

mt19937 gen(rand());
size_t nb, d0, k, efq, num_thread; // number of vectors, dimension

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
BoosterHandle handle; // LightGBM model

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
    if (percentage == 0) {
        part2 = queries_vectors;
        ids2.resize(n);
        std::iota(ids2.begin(), ids2.end(), (id_t)0);
    } else if (percentage == 100) {
        part1 = queries_vectors;
        ids1.resize(n);
        std::iota(ids1.begin(), ids1.end(), (id_t)0);
    } else {
        hnsw.SearchHNNS(utils::Nest(std::move(queries_vectors), queries_vectors.size() / d0, d0), 
            k, efq, knn_all, dist_all, qids_all, 0);
        hnsw.GetComparisonAndClear();
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
    }
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
        LGBM_BoosterPredictForMat(handle, queries_vectors.data(), C_API_DTYPE_FLOAT32, 
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
    }
    partition_timer.Stop();
    std::cout << "[Partition][HNNS] Partition time: " << partition_timer.GetTime() << std::endl;
    std::cout << "[Partition][HNNS] Part1: " << part1.size() / dim << ", part2: " << part2.size() / dim << std::endl;
    return std::make_tuple(part1.size() / dim, part2.size() / dim);
}

int main(int argc, char** argv) 
{
    std::vector<data_t> base_vectors, queries_vectors, train_vectors;
    std::vector<id_t> query_gt, train_gt;
    // std::string dataset = "gist1m";
    // std::string dataset = "imagenet";
    // std::string dataset = "wikipedia";
    std::string dataset = std::string(argv[1]);
    size_t M = std::stol(argv[2]);
    efq = std::stol(argv[3]);
    k = std::stol(argv[4]);
    size_t efc = efq;
    num_thread = std::stol(argv[5]);
    size_t check_stamp = std::stol(argv[6]);
    size_t threshold = std::stol(argv[7]);
    std::string method = std::string(argv[8]);
    std::string base_vectors_path;
    std::string test_vectors_path;
    std::string test_gt_path;
    std::string train_vectors_path;
    std::string train_gt_path;
    faiss::MetricType metric_gpu;
    if (dataset == "imagenet" || dataset == "wikipedia" 
        || dataset == "datacomp-image" || dataset == "datacomp-text") {
        if (dataset == "datacomp-image") {
            base_vectors_path = prefix + "anns/dataset/" + dataset + "/base.i.norm.fvecs";
            test_vectors_path = prefix + "anns/query/" + "datacomp-text" + "/query.t.norm.fvecs";
            train_vectors_path = prefix + "anns/dataset/" + "datacomp-text" + "/learn.t.norm.fvecs";
            test_gt_path = prefix + "anns/query/" + dataset + "/query.t2i.norm.gt.ivecs.cpu.1000";
            train_gt_path = prefix + "anns/dataset/" + dataset + "/learn.t2i.norm.gt.ivecs.cpu.1000";
        } else {
            base_vectors_path = prefix + "anns/dataset/" + dataset + "/base.norm.fvecs";
            test_vectors_path = prefix + "anns/query/" + dataset + "/query.norm.fvecs";
            test_gt_path = prefix + "anns/query/" + dataset + "/query.norm.gt.ivecs.cpu.1000";
            train_vectors_path = prefix + "anns/dataset/" + dataset + "/learn.norm.fvecs";
            train_gt_path = prefix + "anns/dataset/" + dataset + "/learn.norm.gt.ivecs.cpu.1000";
        }
        metric_gpu = faiss::MetricType::METRIC_INNER_PRODUCT;
        metric_cpu = InnerProduct;
    } else {
        base_vectors_path = prefix + "anns/dataset/" + dataset + "/base.fvecs";
        test_vectors_path = prefix + "anns/query/" + dataset + "/query.fvecs";
        test_gt_path = prefix + "anns/query/" + dataset + "/query.gt.ivecs.cpu.1000";
        train_vectors_path = prefix + "anns/dataset/" + dataset + "/learn.fvecs";
        train_gt_path = prefix + "anns/dataset/" + dataset + "/learn.gt.ivecs.cpu.1000";
        metric_gpu = faiss::MetricType::METRIC_L2;
        metric_cpu = L2;
    }

    std::tie(nb, d0) = utils::LoadFromFile(base_vectors, base_vectors_path);
    auto [nq, d1] = utils::LoadFromFile(queries_vectors, test_vectors_path);
    auto [nt, dt] = utils::LoadFromFile(train_vectors, train_vectors_path);
    auto [nbg, dbg] = utils::LoadFromFile(query_gt, test_gt_path);
    auto [ntg, dtg] = utils::LoadFromFile(train_gt, train_gt_path);

    auto nest_test_vectors = utils::Nest(std::move(queries_vectors), nq, d1);
    auto nest_train_vectors = utils::Nest(std::move(train_vectors), nt, dt);

    base_vectors.resize(nb * d0);
    nb = base_vectors.size() / d0;

    nest_test_vectors.resize(nq);
    nq = nest_test_vectors.size();

    nest_train_vectors.resize(nt / 1);
    nt = nest_train_vectors.size();

    dbg = 1000;
    dtg = 1000;
    nbg = query_gt.size() / dbg;
    ntg = train_gt.size() / dtg;
    
    cout << "Load Data Done!" << endl;

    cout << "Base Vectors: " << nb << endl;
    cout << "Queries Vectors: " << nq << endl;
    cout << "Train Vectors: " << nt << endl;

    cout << "Dimension base_vector: " << d0 << endl;
    cout << "Dimension query_vector: " << d1 << endl;
    cout << "Dimension train_vector: " << dt << endl;

    size_t num_check = 100;
    size_t ef_construction = 1000;
    std::string index_path = 
        // "../index/" + dataset + "."
        "/data/disk1/liuchengjun/HNNS/index/" + dataset + "."
        "M_" + to_string(M) + "." 
        "efc_" + to_string(ef_construction) + ".hnsw";
    
    std::string model_path = "/data/disk1/liuchengjun/HNNS/checkpoint/" + dataset + 
        ".M_" + std::to_string(M) + ".efc_" + std::to_string(efc) + 
        ".efs_" + std::to_string(efc) + 
        ".ck_ts_" + std::to_string(check_stamp) + 
        ".ncheck_100.recall@1000.thr_" + std::to_string(threshold);
    if (method == "hnns_qonly") model_path += ".qonly";
    model_path += ".txt";

    std::vector<data_t> test_vector_cpu, test_vector_gpu;
    std::vector<id_t> test_ids_cpu, test_ids_gpu;

    utils::Timer e2e_timer, hnsw_timer;
    std::cout << "dataset: " << dataset << std::endl;

    std::vector<faiss::idx_t> knn_gpu(nq * k_gpu);
    std::vector<data_t> dist_gpu(nq * k_gpu);

    std::thread gpu_thread;
    int device = 6;
    faiss::gpu::StandardGpuResources res;
    faiss::gpu::GpuIndexFlatConfig config;
    config.device = device;
    faiss::gpu::GpuIndexFlat gpu_index(&res, d0, metric_gpu, config);
    gpu_thread = std::thread([&gpu_index, &base_vectors] {
        gpu_index.add(nb, base_vectors.data());
    });

    auto hnsw = std::make_unique<anns::graph::HNSW<data_t>> (
        base_vectors, index_path, dataset,
        k, check_stamp, metric_cpu);
    hnsw->SetNumThreads(num_thread);
    gpu_thread.join();

    std::cout << "[LightGBM] model_path: " << model_path << std::endl;
    int out_num_iterations, result;
    result = LGBM_BoosterCreateFromModelfile(model_path.data(), &out_num_iterations, &handle);
    if (result == 0) {
        hnsw->LoadLightGBM(handle);
        std::cout << "[LightGBM] Model loaded successfully." << std::endl;
    } else {
        std::cout << "[LightGBM] Failed to load model." << std::endl;
    }

    std::vector<std::vector<id_t>> knn_cpu, knn_all;
    std::vector<std::vector<data_t>> dist_cpu, dist_all;
    qids_all.resize(nq);
    std::iota(qids_all.begin(), qids_all.end(), (id_t)0);
    hnsw->GetComparisonAndClear();
    size_t nq_cpu, nq_gpu;
    std::vector<int> pcts = {
        0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50
        // , 55, 60, 65, 
        // 70, 72, 74, 76, 78, 
        // 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 
        // 95, 96, 97, 98, 99, 100
    };
    for (int pct = 51; pct <= 100; pct += 1) {
        pcts.push_back(pct);
    }
    std::reverse(pcts.begin(), pcts.end());
    
    // for (int pct = 0; pct <= 100; pct += 5) {
    for (auto pct : pcts) {
        for (int iter = 0; iter < 3; ++iter) {

            // Random
            if (method == "random") {
                e2e_timer.Reset();    e2e_timer.Start();
                std::tie(nq_cpu, nq_gpu) = partition_random(queries_vectors, test_vector_cpu, test_vector_gpu, test_ids_cpu, test_ids_gpu, d0, pct);
                knn_gpu.resize(nq_gpu * k_gpu);
                dist_gpu.resize(nq_gpu * k_gpu);

                if (nq_gpu > 0) {
                    gpu_thread = std::thread([&gpu_index, &test_vector_gpu, &knn_gpu, &dist_gpu, &nq_gpu] {
                        gpu_index.search(nq_gpu, test_vector_gpu.data(), k_gpu, dist_gpu.data(), knn_gpu.data());
                    });
                }
                hnsw_timer.Reset();    hnsw_timer.Start();
                hnsw->Search(utils::Nest(std::move(test_vector_cpu), test_vector_cpu.size() / d1, d1), 
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
                std::tie(nq_cpu, nq_gpu) = partition_hnns(queries_vectors, test_vector_cpu, test_vector_gpu, test_ids_cpu, test_ids_gpu, d0, *hnsw, pct);
                knn_gpu.resize(nq_gpu * k_gpu);
                dist_gpu.resize(nq_gpu * k_gpu);

                if (nq_gpu > 0) {
                    gpu_thread = std::thread([&gpu_index, &test_vector_gpu, &knn_gpu, &dist_gpu, &nq_gpu] {
                        gpu_index.search(nq_gpu, test_vector_gpu.data(), k_gpu, dist_gpu.data(), knn_gpu.data());
                    });
                }
                hnsw->SearchHNNS(utils::Nest(std::move(test_vector_cpu), test_vector_cpu.size() / d1, d1), 
                    k, efq, knn_cpu, dist_cpu, test_ids_cpu, 1);
                // hnsw->Search(utils::Nest(std::move(test_vector_cpu), test_vector_cpu.size() / d1, d1), 
                //     k, efq, knn_cpu, dist_cpu);
                if (nq_gpu > 0) {
                    gpu_thread.join();
                }
                e2e_timer.Stop();
                hnsw->Reset();
                std::cout << "[Query][HNNS] E2E time: " << e2e_timer.GetTime() << std::endl;
            } else if (method == "hnns_qonly") {
                knn_all.clear();         dist_all.clear();
                e2e_timer.Reset();    e2e_timer.Start();
                std::tie(nq_cpu, nq_gpu) = partition_hnns_qonly(queries_vectors, test_vector_cpu, test_vector_gpu, test_ids_cpu, test_ids_gpu, d0, *hnsw, pct);
                knn_gpu.resize(nq_gpu * k_gpu);
                dist_gpu.resize(nq_gpu * k_gpu);

                if (nq_gpu > 0) {
                    gpu_thread = std::thread([&gpu_index, &test_vector_gpu, &knn_gpu, &dist_gpu, &nq_gpu] {
                        gpu_index.search(nq_gpu, test_vector_gpu.data(), k_gpu, dist_gpu.data(), knn_gpu.data());
                    });
                }
                hnsw_timer.Reset();    hnsw_timer.Start();
                hnsw->Search(utils::Nest(std::move(test_vector_cpu), test_vector_cpu.size() / d1, d1), 
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
                num_recall_cpu += utils::GetRecallCount(ck, dbg, query_gt, knn_cpu[i], test_ids_cpu[i]);
            }
            for (int i = 0; i < test_ids_gpu.size(); ++i) {
                num_recall_gpu += utils::GetRecallCount(ck, dbg, query_gt, nested_knn_gpu[i], test_ids_gpu[i]);
            }
            num_recall = num_recall_cpu + num_recall_gpu;
            // std::cout << "[Query][CPU] Recall@" << ck << ": " << (double)num_recall_cpu / test_ids_cpu.size() / ck << std::endl;
            // std::cout << "[Query][GPU] Recall@" << ck << ": " << (double)num_recall_gpu / test_ids_gpu.size() / ck << std::endl;
            std::string pct_str = "CPU" + std::to_string(pct) + "% + ""GPU" + std::to_string(100 - pct) + "%";
            std::cout << "[Query][HNNS = " << pct_str << "] Recall@" << ck << ": " << (double)num_recall / nq / ck << std::endl;
            std::cout << "[Query][HNSW] avg comparison: " << hnsw->GetComparisonAndClear() / (double)(test_vector_cpu.size() / d1) << std::endl;
        }
    }
    return 0;
}
// nohup ./hnns datacomp-comp 32 1000 1000 48 1000 hnns > ../log/datacomp-image.M_32.efc_1000.efs_1000.ck_ts_1000.ncheck_100.recall@1000.nthread_48.hnns_optimized.log &