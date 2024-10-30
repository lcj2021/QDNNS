#include <string>
#include <vector>
#include "graph/hnsw.hpp"
#include "utils/resize.hpp"
#include "utils/timer.hpp"
#include "utils/recall.hpp"
#include "distance.hpp"

using data_t = float;
using id_t = uint32_t;
using namespace std;

std::string prefix = "/home/zhengweiguo/liuchengjun/";
std::string idx_prefix = "/data/disk1/liuchengjun/HNNS/checkpoint/";
float (*metric)(const data_t *, const data_t *, size_t) = nullptr;

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
    if (dataset == "imagenet" || dataset == "wikipedia" 
        || dataset == "datacomp-image" || dataset == "datacomp-text") {
        base_vectors_path = prefix + "anns/dataset/" + dataset + "/base.norm.fvecs";
        test_vectors_path = prefix + "anns/query/" + dataset + "/query.norm.fvecs";
        test_gt_path = prefix + "anns/query/" + dataset + "/query.norm.gt.ivecs.cpu.1000";
        train_vectors_path = prefix + "anns/dataset/" + dataset + "/learn.norm.fvecs";
        train_gt_path = prefix + "anns/dataset/" + dataset + "/learn.norm.gt.ivecs.cpu.1000";
        metric = InnerProduct;
    } else {
        base_vectors_path = prefix + "anns/dataset/" + dataset + "/base.fvecs";
        test_vectors_path = prefix + "anns/query/" + dataset + "/query.fvecs";
        test_gt_path = prefix + "anns/query/" + dataset + "/query.gt.ivecs.cpu.1000";
        train_vectors_path = prefix + "anns/dataset/" + dataset + "/learn.fvecs";
        train_gt_path = prefix + "anns/dataset/" + dataset + "/learn.gt.ivecs.cpu.1000";
        metric = L2;
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

    dbg = 1000;
    dtg = 1000;
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

    size_t check_stamp = 2000;
    size_t num_check = 100;

    utils::Timer build_timer, query_timer;
    size_t ef_construction = 1000;
    std::string index_path = 
        // "../index/" + dataset + "."
        "/data/disk1/liuchengjun/HNNS/index/" + dataset + "."
        "M_" + to_string(M) + "." 
        "efc_" + to_string(ef_construction) + ".hnsw";
    std::string idx_path = idx_prefix + dataset + "."
            "M_" + std::to_string(M) + "." 
            "efc_" + std::to_string(ef_construction) + "."
            "efs_" + std::to_string(efq) + "."
            "ck_ts_" + std::to_string(check_stamp) + "."
            "ncheck_" + std::to_string(num_check) + "."
            "recall@" + std::to_string(k) + "."
            "thr_" + std::to_string(threshold) + ".hnns_cpu_idx.ivecs";
    // std::vector<id_t> cpu_idx;
    // utils::LoadFromFile(cpu_idx, idx_path);
    // nq = std::accumulate(cpu_idx.begin(), cpu_idx.end(), 0);

    // std::vector<std::vector<data_t>> nest_test_vectors_cpu(nq);
    // for (int i = 0, j = 0; i < nest_test_vectors.size(); ++i) {
    //     if (cpu_idx[i] == 1) {
    //         nest_test_vectors_cpu[j ++] = nest_test_vectors[i];
    //     }
    // }
    // std::swap(nest_test_vectors, nest_test_vectors_cpu);


    std::cout << "dataset: " << dataset << std::endl;
    std::cout << "efSearch: " << efq << std::endl;
    std::cout << "efConstruct: " << ef_construction << std::endl;
    std::cout << "M: " << M << std::endl;
    std::cout << "index_path: " << index_path << std::endl;

    auto hnsw = std::make_unique<anns::graph::HNSW<data_t>> (
        base_vectors, index_path, dataset,
        k, check_stamp, metric);
    hnsw->SetNumThreads(96);

    build_timer.Start();
    // hnsw->BuildIndex(base_vectors);
    // hnsw->Save(index_path);
    build_timer.Stop();

    std::vector<std::vector<id_t>> knn;
    std::vector<std::vector<data_t>> dists;
    hnsw->GetComparisonAndClear();
    std::cout << "Build Time: " << build_timer.GetTime() << std::endl;
    
    query_timer.Reset();
    query_timer.Start();
    hnsw->Search(nest_test_vectors, k, efq, knn, dists);
    query_timer.Stop();
    std::cout << "[Query][HNSW] Params: " << "M: " << M << ", efSearch: " << efq << std::endl;
    std::cout << "[Query][HNSW] Using GT from file: " << test_gt_path << std::endl;
    std::cout << "[Query][HNSW] Search time: " << query_timer.GetTime() << std::endl;
    // std::cout << "[Query][HNSW] Recall@" << k << ": " << utils::GetRecall(k, dbg, query_gt, knn) << std::endl;
    for (int ck = 1; ck <= k; ck *= 10) {
        std::cout << "[Query][HNSW] Recall@" << ck << ": " << utils::GetRecall(ck, dbg, query_gt, knn) << std::endl;
    }
    std::cout << "[Query][HNSW] avg comparison: " << hnsw->GetComparisonAndClear() / (double)nq << std::endl;
    // size_t num_recall = 0;
    // for (int i = 0, j = 0; i < cpu_idx.size(); ++i) {
    //     if (cpu_idx[i] == 1) {
    //         num_recall += utils::GetRecallCount(k, dbg, query_gt, knn[j++], i);
    //     }
    // }
    // std::cout << "[Query][HNSW] avg recall: " << num_recall << ", " << num_recall / (double)nq << std::endl;

    query_timer.Reset();
    query_timer.Start();
    hnsw->Search(nest_train_vectors, k, efq, knn, dists);
    query_timer.Stop();
    std::cout << "[Train][HNSW] Params: " << "M: " << M << ", efSearch: " << efq << std::endl;
    std::cout << "[Train][HNSW] Using GT from file: " << train_gt_path << std::endl;
    std::cout << "[Train][HNSW] Search time: " << query_timer.GetTime() << std::endl;
    for (int ck = 1; ck <= k; ck *= 10) {
        std::cout << "[Train][HNSW] Recall@" << ck << ": " << utils::GetRecall(ck, dtg, train_gt, knn) << std::endl;
    }
    std::cout << "[Train][HNSW] avg comparison: " << hnsw->GetComparisonAndClear() / (double)nt << std::endl;
    return 0;
}

// ./hnsw_run imagenet 96 3000 1000 1000
// ./hnsw_run wikipedia 128 3000 1000 1000
// ./hnsw_run datacomp-image 48 1000 1000 1000
// ./hnsw_run deep100m 32 1000 1000 1000