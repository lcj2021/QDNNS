#include <tuple>
#include <vector>
#include <sys/stat.h>
#include <utils/binary_io.hpp>

using data_t = float;
using id_t = uint32_t;

namespace utils
{
    std::vector<std::string> split(std::string str, char delimiter) 
    {
        std::vector<std::string> tokens;
        size_t start = 0;
        size_t end = 0;
        while (end < str.length()) {
            if (str[end] == delimiter) {
                if (end > start) {
                    tokens.push_back(str.substr(start, end - start));
                }
                start = end + 1;
            }
            end++;
        }
        if (start < str.length()) {
            tokens.push_back(str.substr(start));
        }
        return tokens;
    }

    inline bool is_exists(const std::string &name)
    {
        struct stat buffer;
        return (stat(name.c_str(), &buffer) == 0);
    }

    struct BaseQueryGtConfig
    {
        std::string base_path, query_path, query_gt_path;
        size_t num_base, num_query, num_query_gt;
        size_t dim_base, dim_query, dim_query_gt;
        int metric;
    };

    struct BaseQueryLearnGtConfig: public BaseQueryGtConfig
    {
        size_t num_learn, num_learn_gt;
        size_t dim_learn, dim_learn_gt;
        std::string learn_path, learn_gt_path;
        BaseQueryLearnGtConfig() {}
        BaseQueryLearnGtConfig(struct BaseQueryGtConfig base_query_gt, 
            size_t num_learn, size_t num_learn_gt, size_t dim_learn, size_t dim_learn_gt,
            std::string learn_path, std::string learn_gt_path):
            BaseQueryGtConfig(base_query_gt), num_learn(num_learn), num_learn_gt(num_learn_gt),
            dim_learn(dim_learn), dim_learn_gt(dim_learn_gt), learn_path(learn_path), learn_gt_path(learn_gt_path) 
        {

        }
    };

    class DataLoader
    {
    public:
        DataLoader(const std::string &base_name, const std::string &query_name) 
        {
            base_tokens = split(base_name, '.');       // datacomp-image.base or imagenet.learn
            assert(base_tokens.size() == 2);
            base_path = base_prefix + base_tokens[0] + "/" + base_tokens[1] + ".";

            if (base_tokens[0] == "imagenet" || base_tokens[0] == "wikipedia" 
                || base_tokens[0] == "datacomp-image" || base_tokens[0] == "datacomp-text"
                || base_tokens[0] == "datacomp-combined")
            {
                if (base_tokens[0].substr(0, 8) == "datacomp") {
                    if (base_tokens[0] == "datacomp-text") {
                        base_path += "t.";
                    } else if (base_tokens[0] == "datacomp-image") {
                        base_path += "i.";
                    } else if (base_tokens[0] == "datacomp-combined") {
                        base_path += "i.t.";
                    }
                }
                base_path += "norm.fvecs";
            } else if (base_tokens[0] == "deep100m") {
                base_path += "fvecs";
            } else if (base_tokens[0] == "spacev100m") {
                base_path += "fvecs";
            }
            std::cout << "[Dataloader] base_path: " << base_path << std::endl;


            query_tokens = split(query_name, '.');     // datacomp-text.learn or imagenet.query
            assert(query_tokens.size() == 2);
            query_path = query_prefix + query_tokens[0] + "/";
            learn_path = learn_prefix + query_tokens[0] + "/";
            query_path += "query.";
            learn_path += "learn.";

            if (query_tokens[0] == "imagenet" || query_tokens[0] == "wikipedia"
                || query_tokens[0] == "datacomp-image" || query_tokens[0] == "datacomp-text"
                || query_tokens[0] == "datacomp-combined") 
            {
                if (query_tokens[0].substr(0, 8) == "datacomp") {
                    if (query_tokens[0] == "datacomp-text") {
                        query_path += "t.";
                        learn_path += "t.";
                    } else if (query_tokens[0] == "datacomp-image") {
                        query_path += "i.";
                        learn_path += "i.";
                    } else if (query_tokens[0] == "datacomp-combined") {
                        query_path += "i.t.";
                        learn_path += "i.t.";
                    }
                }
                query_path += "norm.fvecs";
                learn_path += "norm.fvecs";
            } else {
                query_path += "fvecs";
                learn_path += "fvecs";
            }
            std::cout << "[Dataloader] query_path: " << query_path << std::endl;
            std::cout << "[Dataloader] learn_path: " << learn_path << std::endl;


            if (base_tokens[0].substr(0, 8) == "datacomp" && query_tokens[0].substr(0, 8) == "datacomp") {
                query_gt_path = query_prefix + query_tokens[0] + "/" + "query" + ".";
                learn_gt_path = query_prefix + query_tokens[0] + "/" + "learn" + ".";
                if (query_tokens[0] == "datacomp-text" && base_tokens[0] == "datacomp-text") {
                    query_gt_path += "t2t.";
                    learn_gt_path += "t2t.";
                } else if (query_tokens[0] == "datacomp-image" && base_tokens[0] == "datacomp-image") {
                    query_gt_path += "i2i.";
                    learn_gt_path += "i2i.";
                } else if (query_tokens[0] == "datacomp-text" && base_tokens[0] == "datacomp-image") {
                    query_gt_path += "t2i.";
                    learn_gt_path += "t2i.";
                } else if (query_tokens[0] == "datacomp-image" && base_tokens[0] == "datacomp-text") {
                    query_gt_path += "i2t.";
                    learn_gt_path += "i2t.";
                }
                query_gt_path += base_tokens[1] + ".norm.gt.ivecs.cpu.1000";
                learn_gt_path += base_tokens[1] + ".norm.gt.ivecs.cpu.1000";
            } else if (base_tokens[0] == query_tokens[0]) {
                query_gt_path = query_prefix + query_tokens[0] + "/" + "query" + ".";
                learn_gt_path = query_prefix + query_tokens[0] + "/" + "learn" + ".";
                if (query_tokens[0] == "imagenet" || query_tokens[0] == "wikipedia") {
                    query_gt_path += "norm.";
                    learn_gt_path += "norm.";
                }
                query_gt_path += "gt.ivecs.cpu.1000";
                learn_gt_path += "gt.ivecs.cpu.1000";
            } else {
                std::cerr << "[Dataloader] Invalid query base pair: " << query_name << " " << base_name << std::endl;
                exit(1);
            }
            std::cout << "[Dataloader] query_gt_path: " << query_gt_path << std::endl;
            std::cout << "[Dataloader] learn_gt_path: " << learn_gt_path << std::endl;

            if (base_tokens[0] == "imagenet" || base_tokens[0] == "wikipedia"
                || base_tokens[0] == "datacomp-image" || base_tokens[0] == "datacomp-text"
                || base_tokens[0] == "datacomp-combined") {
                metric = 0;
            } else {
                metric = 1;
            }
        }

        std::tuple<std::vector<data_t>&, std::vector<data_t>&, std::vector<id_t>&,
            BaseQueryGtConfig>
        load()
        {
            std::tie(num_base, dim_base) = utils::LoadFromFile(base_data, base_path);
            std::tie(num_query, dim_query) = utils::LoadFromFile(query_data, query_path);
            std::cout << "[Dataloader] base_data: " << num_base << " x " << dim_base << std::endl;
            std::cout << "[Dataloader] query_data: " << num_query << " x " << dim_query << std::endl;

            if (is_exists(query_gt_path)) {
                std::tie(num_query_gt, dim_query_gt) = utils::LoadFromFile(query_gt_data, query_gt_path);
                assert(num_query_gt * dim_query_gt == num_query * 1000);
                num_query_gt = num_query;
                dim_query_gt = 1000;
                std::cout << "[Dataloader] query_gt_data: " << num_query_gt << " x " << dim_query_gt << std::endl;
            } else {
                std::cout << "[Dataloader] gt not found, please check the path" << std::endl;
            }

            BaseQueryGtConfig config = {
                base_path, query_path, query_gt_path,
                num_base, num_query, num_query_gt, 
                dim_base, dim_query, dim_query_gt, 
                metric
            };
            return {base_data, query_data, query_gt_data, config};
        }

        std::tuple<std::vector<data_t>&, std::vector<data_t>&, std::vector<id_t>&, std::vector<data_t>&, std::vector<id_t>&, 
            BaseQueryLearnGtConfig>
        load_with_learn()
        {
            const auto& [base_data, query_data, query_gt_data, config] = load();

            std::tie(num_learn, dim_learn) = utils::LoadFromFile(learn_data, learn_path);
            std::cout << "[Dataloader] learn_data: " << num_learn << " x " << dim_learn << std::endl;

            if (is_exists(learn_gt_path)) {
                std::tie(num_learn_gt, dim_learn_gt) = utils::LoadFromFile(learn_gt_data, learn_gt_path);
                assert(num_learn_gt * dim_learn_gt == num_learn * 1000);
                num_learn_gt = num_learn;
                dim_learn_gt = 1000;
                std::cout << "[Dataloader] learn_gt_data: " << num_learn_gt << " x " << dim_learn_gt << std::endl;
            } else {
                std::cout << "[Dataloader] gt not found, please check the path" << std::endl;
            }


            BaseQueryLearnGtConfig config_with_learn = {
                config, 
                num_learn, num_learn_gt,
                dim_learn, dim_learn_gt,
                learn_path, learn_gt_path
            };
            return {base_data, query_data, query_gt_data, learn_data, learn_gt_data, config_with_learn};
        }

    private:
        std::string base_prefix = "/home/zhengweiguo/liuchengjun/anns/dataset/";
        std::string query_prefix = "/home/zhengweiguo/liuchengjun/anns/query/";
        std::string learn_prefix = "/home/zhengweiguo/liuchengjun/anns/query/";

        std::string base_path, query_path, learn_path, query_gt_path, learn_gt_path;
        std::vector<std::string> base_tokens, query_tokens, learn_tokens;

        size_t num_base, num_query, num_query_gt, num_learn, num_learn_gt;
        size_t dim_base, dim_query, dim_query_gt, dim_learn, dim_learn_gt;

        std::vector<data_t> base_data, query_data, learn_data;
        std::vector<id_t> query_gt_data, learn_gt_data;

        int metric = 0;
    };
} // namespace utils