#include <GpuIndexFlat.h>
#include <StandardGpuResources.h>
#include <IndexUtils.h>
#include <DeviceUtils.h>
#include <vector>

#include "../utils/binary_io.hpp"
#include "../utils/resize.hpp"
#include "../utils/stimer.hpp"
#include "../utils/recall.hpp"

using data_t = float;
using id_t = uint32_t;
using namespace std;

std::string prefix = "/home/zhengweiguo/liuchengjun/";

int main(int argc, char** argv) 
{
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

    nest_train_vectors.resize(nt);
    nt = nest_train_vectors.size();

    dbg = dtg = 1000;
    nbg = query_gt.size() / dbg;
    ntg = train_gt.size() / dtg;
    
    cout << "Load Data Done!" << endl;

    cout << "Base Vectors: " << nb << endl;
    cout << "Queries Vectors: " << nq << endl;
    cout << "Train Vectors: " << nt << endl;

    cout << "Dimension base_vector: " << d0 << endl;
    cout << "Dimension query_vector: " << d1 << endl;
    cout << "Dimension train_vector: " << dt << endl;

    size_t k = 1;

    utils::STimer query_timer, train_timer;
    std::cout << "dataset: " << dataset << std::endl;

    std::vector<faiss::idx_t> knn(nq * k);
    std::vector<data_t> dist(nq * k);

    int device = 1;
    faiss::gpu::StandardGpuResources res;
    faiss::gpu::GpuIndexFlatConfig config;
    config.device = device;
    faiss::gpu::GpuIndexFlat gpu_index(&res, d0, metric, config);
    gpu_index.add(nb, base_vectors.data());

    query_timer.Reset();    query_timer.Start();
    gpu_index.search(nq, queries_vectors.data(), k, dist.data(), knn.data());
    query_timer.Stop();
    std::cout << "[Query][GPU] Using GT from file: " << test_gt_path << std::endl;
    std::cout << "[Query][GPU] Search time: " << query_timer.GetTime() << std::endl;
    std::cout << "[Query][GPU] Recall@" << k << ": " << utils::GetRecall(k, dbg, query_gt, utils::Nest(knn, nq, k)) << std::endl;

    knn.resize(nt * k);
    dist.resize(nt * k);
    query_timer.Reset();    query_timer.Start();
    gpu_index.search(nt, train_vectors.data(), k, dist.data(), knn.data());
    query_timer.Stop();
    std::cout << "[Train][GPU] Using GT from file: " << train_gt_path << std::endl;
    std::cout << "[Train][GPU] Search time: " << query_timer.GetTime() << std::endl;
    std::cout << "[Train][GPU] Recall@" << k << ": " << utils::GetRecall(k, dtg, train_gt, utils::Nest(knn, nt, k)) << std::endl;
    return 0;
}
