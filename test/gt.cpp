#include <algorithm>
#include "utils/binary_io.hpp"
#include "utils/resize.hpp"
#include "utils/stimer.hpp"
#include "utils/recall.hpp"
#include "flat/IndexFlat.hpp"
#include "distance.hpp"

using data_t = float;
using id_t = uint32_t;
using namespace std;

std::string prefix = "/home/zhengweiguo/liuchengjun/";

int main(int argc, char** argv) 
{
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
    float (*distance)(const data_t *, const data_t *, size_t) = nullptr;
    if (dataset == "imagenet" || dataset == "wikipedia") {
        base_vectors_path = prefix + "anns/dataset/" + dataset + "/base.norm.fvecs";
        test_vectors_path = prefix + "anns/query/" + dataset + "/query.norm.fvecs";
        test_gt_path = prefix + "anns/query/" + dataset + "/query.norm.gt.ivecs.cpu";
        train_vectors_path = prefix + "anns/dataset/" + dataset + "/learn.norm.fvecs";
        train_gt_path = prefix + "anns/dataset/" + dataset + "/learn.norm.gt.ivecs.cpu";
        distance = InnerProduct;
    } else {
        base_vectors_path = prefix + "anns/dataset/" + dataset + "/base.fvecs";
        test_vectors_path = prefix + "anns/query/" + dataset + "/query.fvecs";
        test_gt_path = prefix + "anns/query/" + dataset + "/query.gt.ivecs.cpu";
        train_vectors_path = prefix + "anns/dataset/" + dataset + "/learn.fvecs";
        train_gt_path = prefix + "anns/dataset/" + dataset + "/learn.gt.ivecs.cpu";
        distance = L2;
    }

    auto [nb, d0] = utils::LoadFromFile(base_vectors, base_vectors_path);
    auto [nq, d1] = utils::LoadFromFile(queries_vectors, test_vectors_path);
    auto [nbg, dbg] = utils::LoadFromFile(query_gt, test_gt_path);
    auto [nt, dt] = utils::LoadFromFile(train_vectors, train_vectors_path);
    auto [ntg, dtg] = utils::LoadFromFile(train_gt, train_gt_path);

    auto nest_test_vectors = utils::Nest(std::move(queries_vectors), nq, d1);
    auto nest_train_vectors = utils::Nest(std::move(train_vectors), nt, dt);

    base_vectors.resize(nb * d0);
    nb = base_vectors.size() / d0;

    nest_test_vectors.resize(nq / 1);
    nq = nest_test_vectors.size();

    nest_train_vectors.resize(nt / 1);
    nt = nest_train_vectors.size();

    dbg = dtg = 100;
    nbg = query_gt.size() / dbg;
    ntg = train_gt.size() / dtg;

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

    size_t k = 1'000;
    size_t num_threads_ = 96;

    utils::STimer query_timer;
    std::cout << "dataset: " << dataset << std::endl;

    std::vector<std::vector<id_t>> knn(nq, std::vector<id_t>(k));
    std::vector<std::vector<data_t>> dist(nq, std::vector<data_t>(k));
    anns::flat::IndexFlat<data_t, L2> index(base_vectors, d0);
    index.SetNumThreads(num_threads_);

    // query_timer.Reset();
    // query_timer.Start();
    // index.Search(nest_test_vectors, k, knn, dist);
    // query_timer.Stop();
    // test_gt_path += ".1000";
    // utils::WriteToFile<id_t>(utils::Flatten(knn), {knn.size() * k, 1}, test_gt_path);
    // std::cout << "[Query][FlatCPU] Writing GT to file: " << test_gt_path << std::endl;
    // std::cout << "[Query][FlatCPU] Search time: " << query_timer.GetTime() << std::endl;

    query_timer.Reset();
    query_timer.Start();
    index.Search(nest_train_vectors, k, knn, dist);
    query_timer.Stop();
    train_gt_path += ".1000";
    utils::WriteToFile<id_t>(utils::Flatten(knn), {knn.size() * k, 1}, train_gt_path);
    std::cout << "[Train][FlatCPU] Writing GT to file: " << train_gt_path << std::endl;
    std::cout << "[Train][FlatCPU] Search time: " << query_timer.GetTime() << std::endl;
    return 0;
}

// g++ gt.cpp -std=c++17 -I ../include/ -Ofast -march=native -mtune=native -lrt -fopenmp  && ./a.out