#include <bits/stdc++.h>
#include <GpuIndexFlat.h>
#include <StandardGpuResources.h>
#include <IndexUtils.h>
#include <TestUtils.h>
#include <DeviceUtils.h>
#include <sstream>
#include <vector>

#include "../utils/binary_io.hpp"
#include "../utils/resize.hpp"
#include "../utils/stimer.hpp"
#include "../utils/get_recall.hpp"
// #include "distance.hpp"

using data_t = float;
using id_t = uint32_t;
using namespace std;

std::string prefix = "/home/zhengweiguo/liuchengjun/";

int main(int argc, char** argv) {
    std::vector<data_t> base_vectors, queries_vectors, train_vectors;
    std::vector<id_t> query_gt, train_gt;
    // std::string dataset = "gist1m";
    // std::string dataset = "imagenet";
    std::string dataset = "wikipedia";
    std::string base_vectors_path;
    std::string test_vectors_path;
    std::string test_gt_path;
    std::string train_vectors_path;
    std::string train_gt_path;
    faiss::MetricType metric;
    if (dataset == "imagenet" || dataset == "wikipedia") {
        base_vectors_path = prefix + "anns/dataset/" + dataset + "/base.norm.fvecs";
        test_vectors_path = prefix + "anns/query/" + dataset + "/query.norm.fvecs";
        test_gt_path = prefix + "anns/query/" + dataset + "/query.norm.gt.ivecs";
        train_vectors_path = prefix + "anns/dataset/" + dataset + "/learn.norm.fvecs";
        train_gt_path = prefix + "anns/dataset/" + dataset + "/learn.norm.gt.ivecs.cpu";
        metric = faiss::MetricType::METRIC_INNER_PRODUCT;
    } else {
        base_vectors_path = prefix + "anns/dataset/" + dataset + "/base.fvecs";
        test_vectors_path = prefix + "anns/query/" + dataset + "/query.fvecs";
        test_gt_path = prefix + "anns/query/" + dataset + "/query.gt.ivecs.new";
        train_vectors_path = prefix + "anns/dataset/" + dataset + "/learn.fvecs";
        train_gt_path = prefix + "anns/dataset/" + dataset + "/learn.gt.ivecs.new";
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

    nest_train_vectors.resize(nt / 1);
    nt = nest_train_vectors.size();

    cout << "Load Data Done!" << endl;

    cout << "Base Vectors: " << nb << endl;
    cout << "Queries Vectors: " << nq << endl;
    cout << "Train Vectors: " << nt << endl;

    cout << "Dimension base_vector: " << d0 << endl;
    cout << "Dimension query_vector: " << d1 << endl;
    cout << "Dimension train_vector: " << dt << endl;

    size_t k = 100;
    size_t num_threads_ = 24;

    utils::STimer query_timer, train_timer;
    std::cout << "dataset: " << dataset << std::endl;

    // std::vector<std::vector<id_t>> knn(nq, std::vector<id_t>(k));
    std::vector<faiss::idx_t> knn(nq * k);
    std::vector<std::vector<id_t>> knn_i32(nq, std::vector<id_t>(k));
    std::vector<data_t> dist(nq * k);

    int device = 1;
    faiss::gpu::StandardGpuResources res;
    faiss::gpu::GpuIndexFlatConfig config;
    config.device = device;
    faiss::gpu::GpuIndexFlat gpuIndex(&res, d0, metric, config);
    gpuIndex.add(nb, base_vectors.data());

    // query_timer.Reset();
    // query_timer.Start();

    // gpuIndex.search(nq, queries_vectors.data(), k, dist.data(), knn.data());

    // query_timer.Stop();
    // std::cout << "[Query GT] Query search time: " << query_timer.GetTime() << std::endl;
    // cout << "Query GT write to file: " << test_gt_path << endl;
    // for (int i = 0; i < nq; ++i) {
    //     for (int j = 0; j < k; ++j) {
    //         knn_i32[i][j] = knn[i * k + j];
    //     }
    // }

    // for (int i = 0; i < 10; ++i) {
    //     for (int j = 0; j < k; ++j) {
    //         std::cout << dist[i * k + j] << " ";
    //     } std::cout << std::endl;
    // }

    // std::cout << "[Naive] Recall@" << k << ": " << utils::GetRecall(k, dbg, query_gt, knn_i32) << std::endl;

    knn.resize(nt * k);
    knn_i32.resize(nt, std::vector<id_t>(k));
    dist.resize(nt * k);
    query_timer.Reset();
    query_timer.Start();

    gpuIndex.search(nt, train_vectors.data(), k, dist.data(), knn.data());

    query_timer.Stop();
    std::cout << "[Train] Train search time: " << query_timer.GetTime() << std::endl;
    cout << "Using Train GT from file: " << train_gt_path << endl;
    for (int i = 0; i < nt; ++i) {
        for (int j = 0; j < k; ++j) {
            knn_i32[i][j] = knn[i * k + j];
        }
    }

    for (int i = 0; i < 10; ++i) {
        for (int j = 0; j < k; ++j) {
            std::cout << dist[i * k + j] << " ";
        } std::cout << std::endl;
    }

    std::cout << "[Train] Recall@" << k << ": " << utils::GetRecall(k, dtg, train_gt, knn_i32) << std::endl;
    cout << "Using Train GT from file: " << test_gt_path << endl;
    return 0;
}
// g++ test_gpu.cpp -std=c++17 -I ../include/ -Lbuild/ -Ofast -march=native -mtune=native -lrt -fopenmp -o test_gpu && ./test_gpu