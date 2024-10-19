#include <bits/stdc++.h>
#include "graph/hnsw.hpp"
#include "utils/resize.hpp"
#include "utils/stimer.hpp"
#include "utils/get_recall.hpp"
#include "distance.hpp"

using data_t = float;
using id_t = uint32_t;
using namespace std;

std::string prefix = "/home/zhengweiguo/liuchengjun/";

int main(int argc, char** argv) {
    std::vector<data_t> base_vectors, queries_vectors, train_vectors;
    std::vector<id_t> query_gt, train_gt;
    std::string dataset = "gist1m";
    // std::string dataset = "imagenet";
    // std::string dataset = "wikipedia";
    std::string base_vectors_path;
    std::string test_vectors_path;
    std::string test_gt_path;
    std::string train_vectors_path;
    std::string train_gt_path;
    if (dataset == "imagenet" || dataset == "wikipedia") {
        base_vectors_path = prefix + "anns/dataset/" + dataset + "/base.norm.fvecs";
        test_vectors_path = prefix + "anns/query/" + dataset + "/query.norm.fvecs";
        test_gt_path = prefix + "anns/query/" + dataset + "/query.norm.gt.ivecs";
        train_vectors_path = prefix + "anns/dataset/" + dataset + "/learn.norm.fvecs";
        train_gt_path = prefix + "anns/dataset/" + dataset + "/learn.norm.gt.ivecs";
    } else {
        base_vectors_path = prefix + "anns/dataset/" + dataset + "/base.fvecs";
        test_vectors_path = prefix + "anns/query/" + dataset + "/query.fvecs";
        test_gt_path = prefix + "anns/query/" + dataset + "/query.gt.ivecs.new";
        train_vectors_path = prefix + "anns/dataset/" + dataset + "/learn.fvecs";
        train_gt_path = prefix + "anns/dataset/" + dataset + "/learn.gt.ivecs.new";
    }

    auto [nb, d0] = utils::LoadFromFile(base_vectors, base_vectors_path);
    auto [nq, d1] = utils::LoadFromFile(queries_vectors, test_vectors_path);
    auto [nbg, dbg] = utils::LoadFromFile(query_gt, test_gt_path);
    auto [nt, dt] = utils::LoadFromFile(train_vectors, train_vectors_path);
    auto [ntg, dtg] = utils::LoadFromFile(train_gt, train_gt_path);

    auto nest_test_vectors = utils::Nest(std::move(queries_vectors), nq, d1);
    auto nest_train_vectors = utils::Nest(std::move(train_vectors), nt, dt);

    base_vectors.resize(nb * d0 / 1);
    nb = base_vectors.size() / d0;

    nest_test_vectors.resize(nq / 1);
    nq = nest_test_vectors.size();

    nest_train_vectors.resize(nt / 1);
    nt = nest_train_vectors.size();

    cout << "Load Data Done!" << endl;

    cout << "Base Vectors: " << nb << endl;
    cout << "Queries Vectors: " << nq << endl;
    cout << "Base GT Vectors: " << nbg << endl;
    cout << "Train GT Vectors: " << ntg << endl;
    cout << "Train Vectors: " << nt << endl;

    cout << "Dimension base_vector: " << d0 << endl;
    cout << "Dimension query_vector: " << d1 << endl;
    cout << "Dimension query GT: " << dbg << endl;
    cout << "Dimension train GT: " << dtg << endl;
    cout << "Dimension train_vector: " << dt << endl;

    size_t efq = 1000;
    size_t k = 10;
    size_t check_stamp = 2000;
    size_t num_clusters = 8192;
    std::cout << "efSearch: " << efq << std::endl;

    utils::STimer build_timer;
    utils::STimer query_timer;
    utils::STimer train_timer;
    size_t M = 64;
    size_t ef_construction = 1000;
    std::string index_path = 
        "../index/" + dataset + "."
        "M_" + to_string(M) + "." 
        "efc_" + to_string(ef_construction) + ".hnsw";
    std::cout << "dataset: " << dataset << std::endl;
    std::cout << "efSearch: " << efq << std::endl;
    std::cout << "efConstruct: " << ef_construction << std::endl;
    std::cout << "M: " << M << std::endl;

    auto hnsw = std::make_unique<anns::graph::HNSW<data_t, L2>> (
        base_vectors, index_path, dataset,
        num_clusters, check_stamp);
    hnsw->SetNumThreads(160);

    // hnsw->dataset = dataset;
    build_timer.Start();
    // hnsw->BuildIndex(base_vectors);
    // hnsw->Save(index_path);
    build_timer.Stop();

    std::vector<std::vector<id_t>> knn;
    std::vector<std::vector<data_t>> dists;
    hnsw->GetComparisonAndClear();
    std::cout << "Build Time: " << build_timer.GetTime() << std::endl;
    
    query_timer.Start();
    hnsw->SearchGetData(nest_test_vectors, k, efq, knn, dists, 1);
    query_timer.Stop();
    std::cout << "Query search time: " << query_timer.GetTime() << std::endl;
    std::cout << "Recall@" << k << ": " << utils::GetRecall(k, dbg, query_gt, knn) << std::endl;
    std::cout << "avg comparison: " << hnsw->GetComparisonAndClear() / (double)nq << std::endl;

    train_timer.Start();
    hnsw->SearchGetData(nest_train_vectors, k, efq, knn, dists, 2);
    train_timer.Stop();
    std::cout << "Train search time: " << train_timer.GetTime() << std::endl;
    std::cout << "Recall@" << k << ": " << utils::GetRecall(k, dtg, train_gt, knn) << std::endl;
    std::cout << "avg comparison: " << hnsw->GetComparisonAndClear() / (double)nt << std::endl;
    hnsw->SaveData();
  // ... ... ...
  return 0;
}

// g++ hnsw_get_data.cpp -std=c++17 -I ../include/ -Ofast -march=native -mtune=native -lrt -fopenmp  && ./a.out