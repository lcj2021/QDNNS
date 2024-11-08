#include <algorithm>
#include "utils/binary_io.hpp"
#include "utils/resize.hpp"
#include "utils/timer.hpp"
#include "utils/recall.hpp"
#include "flat/IndexFlat.hpp"
#include "distance.hpp"

using data_t = float;
using id_t = uint32_t;
using namespace std;

std::string prefix = "/home/zhengweiguo/liuchengjun/";
float (*metric)(const data_t *, const data_t *, size_t) = nullptr;

int main(int argc, char** argv) 
{
    std::vector<data_t> base_vectors, queries_vectors, train_vectors;
    // std::string dataset = "gist1m";
    // std::string dataset = "imagenet";
    // std::string dataset = "wikipedia";
    // std::string dataset = "datacomp-image";
    // std::string dataset = "deep100m";
    std::string dataset = std::string(argv[1]);
    std::string base_vectors_path;
    std::string test_vectors_path;
    std::string test_gt_path;
    std::string train_vectors_path;
    std::string train_gt_path;
    if (dataset == "imagenet" || dataset == "wikipedia" 
        || dataset == "datacomp-image" || dataset == "datacomp-text") {
        if (dataset == "datacomp-image" ) {
            base_vectors_path = prefix + "anns/dataset/" + dataset + "/base.i.norm.fvecs";
            test_vectors_path = prefix + "anns/query/" + "datacomp-text" + "/query.t.norm.fvecs";
            train_vectors_path = prefix + "anns/dataset/" + "datacomp-text" + "/learn.t.norm.fvecs";
            test_gt_path = prefix + "anns/query/" + dataset + "/query.t2i.norm.gt.ivecs.cpu.1000";
            train_gt_path = prefix + "anns/dataset/" + dataset + "/learn.norm.t2i.gt.ivecs.cpu.1000";
        } else {
            base_vectors_path = prefix + "anns/dataset/" + dataset + "/base.norm.fvecs";
            test_vectors_path = prefix + "anns/query/" + dataset + "/query.norm.fvecs";
            test_gt_path = prefix + "anns/query/" + dataset + "/query.norm.gt.ivecs.cpu.1000";
            train_vectors_path = prefix + "anns/dataset/" + dataset + "/learn.norm.fvecs";
            train_gt_path = prefix + "anns/dataset/" + dataset + "/learn.norm.gt.ivecs.cpu.1000";
        }
        metric = InnerProduct;
    } else {
        base_vectors_path = prefix + "anns/dataset/" + dataset + "/base.fvecs";
        test_vectors_path = prefix + "anns/query/" + dataset + "/query.fvecs";
        test_gt_path = prefix + "anns/query/" + dataset + "/query.gt.ivecs.cpu";
        train_vectors_path = prefix + "anns/dataset/" + dataset + "/learn.fvecs";
        train_gt_path = prefix + "anns/dataset/" + dataset + "/learn.gt.ivecs.cpu";
        metric = L2;
    }

    auto [nb, d0] = utils::LoadFromFile(base_vectors, base_vectors_path);
    auto [nq, d1] = utils::LoadFromFile(queries_vectors, test_vectors_path);
    auto [nt, dt] = utils::LoadFromFile(train_vectors, train_vectors_path);

    auto nest_test_vectors = utils::Nest(std::move(queries_vectors), nq, d1);
    auto nest_train_vectors = utils::Nest(std::move(train_vectors), nt, dt);

    base_vectors.resize(nb * d0);
    nb = base_vectors.size() / d0;

    nest_test_vectors.resize(nq / 1);
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

    size_t k = 1'000;
    size_t num_threads_ = 96;

    utils::Timer query_timer;
    std::cout << "dataset: " << dataset << std::endl;

    std::vector<std::vector<id_t>> knn(nq, std::vector<id_t>(k));
    std::vector<std::vector<data_t>> dist(nq, std::vector<data_t>(k));
    anns::flat::IndexFlat<data_t> index(base_vectors, d0, metric);
    index.SetNumThreads(num_threads_);

    query_timer.Reset();
    query_timer.Start();
    index.Search(nest_train_vectors, k, knn, dist);
    query_timer.Stop();
    utils::WriteToFile<id_t>(utils::Flatten(knn), {knn.size() * k, 1}, train_gt_path);
    std::cout << "[Train][FlatCPU] Writing GT to file: " << train_gt_path << std::endl;
    std::cout << "[Train][FlatCPU] Search time: " << query_timer.GetTime() << std::endl;

    query_timer.Reset();
    query_timer.Start();
    index.Search(nest_test_vectors, k, knn, dist);
    query_timer.Stop();
    utils::WriteToFile<id_t>(utils::Flatten(knn), {knn.size() * k, 1}, test_gt_path);
    std::cout << "[Query][FlatCPU] Writing GT to file: " << test_gt_path << std::endl;
    std::cout << "[Query][FlatCPU] Search time: " << query_timer.GetTime() << std::endl;

    return 0;
}
