#include <GpuIndexFlat.h>
#include <StandardGpuResources.h>
#include <IndexUtils.h>
#include <DeviceUtils.h>
#include <vector>
#include <numeric>

#include "../utils/binary_io.hpp"
#include "../utils/resize.hpp"
#include "../utils/timer.hpp"
#include "../utils/recall.hpp"

using data_t = float;
using id_t = uint32_t;
using namespace std;

std::string prefix = "/home/zhengweiguo/liuchengjun/";
std::string idx_prefix = "/data/disk1/liuchengjun/HNNS/checkpoint/";

int main(int argc, char** argv) 
{
    std::vector<data_t> base_vectors, queries_vectors, train_vectors;
    std::vector<id_t> query_gt, train_gt;
    // std::string dataset = "gist1m";
    // std::string dataset = "imagenet";
    // std::string dataset = "wikipedia";
    std::string dataset = std::string(argv[1]);
    size_t M = std::stol(argv[2]);
    size_t efq = std::stol(argv[3]);
    size_t k = std::stol(argv[4]);
    size_t threshold = std::stol(argv[5]);
    std::string base_vectors_path;
    std::string test_vectors_path;
    std::string test_gt_path;
    std::string train_vectors_path;
    std::string train_gt_path;
    faiss::MetricType metric;
    if (dataset == "imagenet" || dataset == "wikipedia" 
        || dataset == "datacomp-image" || dataset == "datacomp-text") {
        base_vectors_path = prefix + "anns/dataset/" + dataset + "/base.norm.fvecs";
        test_vectors_path = prefix + "anns/query/" + dataset + "/query.norm.fvecs";
        test_gt_path = prefix + "anns/query/" + dataset + "/query.norm.gt.ivecs.cpu.1000";
        train_vectors_path = prefix + "anns/dataset/" + dataset + "/learn.norm.fvecs";
        train_gt_path = prefix + "anns/dataset/" + dataset + "/learn.norm.gt.ivecs.cpu.1000";
        metric = faiss::MetricType::METRIC_INNER_PRODUCT;
    } else {
        base_vectors_path = prefix + "anns/dataset/" + dataset + "/base.fvecs";
        test_vectors_path = prefix + "anns/query/" + dataset + "/query.fvecs";
        test_gt_path = prefix + "anns/query/" + dataset + "/query.gt.ivecs.cpu.1000";
        train_vectors_path = prefix + "anns/dataset/" + dataset + "/learn.fvecs";
        train_gt_path = prefix + "anns/dataset/" + dataset + "/learn.gt.ivecs.cpu.1000";
        metric = faiss::MetricType::METRIC_L2;
    }

    auto [nb, d0] = utils::LoadFromFile(base_vectors, base_vectors_path);
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

    nest_train_vectors.resize(nt / 4);
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

    size_t check_stamp = 2000;
    size_t num_check = 100;
    size_t ef_construction = 1000;

    std::string idx_path = idx_prefix + dataset + "."
            "M_" + std::to_string(M) + "." 
            "efc_" + std::to_string(ef_construction) + "."
            "efs_" + std::to_string(efq) + "."
            "ck_ts_" + std::to_string(check_stamp) + "."
            "ncheck_" + std::to_string(num_check) + "."
            "recall@" + std::to_string(1000) + "."
            "thr_" + std::to_string(threshold) + ".hnns_gpu_idx.ivecs";
    // std::vector<id_t> gpu_idx;
    // utils::LoadFromFile(gpu_idx, idx_path);

    // std::vector<std::vector<data_t>> nest_test_vectors_gpu;
    // for (int i = 0; i < nest_test_vectors.size(); ++i) {
    //     // if (gpu_idx[i] == 0) continue;
    //     nest_test_vectors_gpu.emplace_back(nest_test_vectors[i]);
    // }
    // auto queries_vectors_gpu = utils::Flatten(nest_test_vectors_gpu);
    // std::swap(nest_test_vectors, nest_test_vectors_gpu);
    // nq = nest_test_vectors_gpu.size();

    utils::Timer query_timer, train_timer;
    std::cout << "dataset: " << dataset << std::endl;

    std::vector<faiss::idx_t> knn(nq * k);
    std::vector<data_t> dist(nq * k);

    int device = 1;
    faiss::gpu::StandardGpuResources res;
    faiss::gpu::GpuIndexFlatConfig config;
    config.device = device;
    faiss::gpu::GpuIndexFlat gpu_index(&res, d0, metric, config);
    gpu_index.add(nb, base_vectors.data());

    while (1) {
        ;
        query_timer.Reset();    query_timer.Start();
        gpu_index.search(nq, queries_vectors.data(), k, dist.data(), knn.data());
        // gpu_index.search(nq, queries_vectors_gpu.data(), k, dist.data(), knn.data());
        query_timer.Stop();
        auto nested_knn = utils::Nest(knn, nq, k);
        std::cout << "[Query][GPU] Using GT from file: " << test_gt_path << std::endl;
        std::cout << "[Query][GPU] Search time: " << query_timer.GetTime() << std::endl;
        for (int ck = 1; ck <= k; ck *= 10) {
            std::cout << "[Query][GPU] Recall@" << ck << ": " << utils::GetRecall(ck, dbg, query_gt, nested_knn) << std::endl;
        }
    }

    // query_timer.Reset();    query_timer.Start();
    // gpu_index.search(nq, queries_vectors.data(), k, dist.data(), knn.data());
    // // gpu_index.search(nq, queries_vectors_gpu.data(), k, dist.data(), knn.data());
    // query_timer.Stop();
    // auto nested_knn = utils::Nest(knn, nq, k);
    // std::cout << "[Query][GPU] Using GT from file: " << test_gt_path << std::endl;
    // std::cout << "[Query][GPU] Search time: " << query_timer.GetTime() << std::endl;
    // for (int ck = 1; ck <= k; ck *= 10) {
    //     std::cout << "[Query][GPU] Recall@" << ck << ": " << utils::GetRecall(ck, dbg, query_gt, nested_knn) << std::endl;
    // }

    // // size_t num_recall = 0;
    // // for (int i = 0, j = 0; i < gpu_idx.size(); ++i) {
    // //     // if (gpu_idx[i] == 0) continue;
    // //     num_recall += utils::GetRecallCount(k, dbg, query_gt, nested_knn[j++], i);
    // // }
    // // std::cout << "nq: " << nq << std::endl;
    // // std::cout << "[Query][HNSW] avg recall: " << num_recall << ", " << num_recall / (double)nq << std::endl;

    // knn.resize(nt * k);
    // dist.resize(nt * k);
    // query_timer.Reset();    query_timer.Start();
    // gpu_index.search(nt, train_vectors.data(), k, dist.data(), knn.data());
    // query_timer.Stop();
    // nested_knn = utils::Nest(knn, nq, k);
    // std::cout << "[Train][GPU] Using GT from file: " << train_gt_path << std::endl;
    // std::cout << "[Train][GPU] Search time: " << query_timer.GetTime() << std::endl;
    // for (int ck = 1; ck <= k; ck *= 10) {
    //     std::cout << "[Train][GPU] Recall@" << ck << ": " << utils::GetRecall(ck, dtg, train_gt, nested_knn) << std::endl;
    // }
    return 0;
}
// ./flat_gpu wikipedia 96 3000 1024 1000