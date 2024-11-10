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
        std::string base_path, query_path, gt_path;
        size_t num_base, num_query, num_gt;
        size_t dim_base, dim_query, dim_gt;
        int metric;
    };

    class DataLoader
    {
    public:
        DataLoader() {}

        std::tuple<std::vector<data_t>, std::vector<data_t>, std::vector<id_t>,
            BaseQueryGtConfig>
        load(const std::string &base_name, const std::string &query_name)
        {
            std::vector<data_t> base_data, query_data;
            std::vector<id_t> gt_data;

            std::string gt_path = "";
            std::string base_path, query_path;
            size_t num_base, num_query, num_gt = 0;
            size_t dim_base, dim_query, dim_gt = 0;

            auto base_tokens = split(base_name, '.');       // datacomp-image.base or imagenet.learn
            assert(base_tokens.size() == 2);
            base_path = base_prefix + base_tokens[0] + "/" + base_tokens[1] + ".";

            if (base_tokens[0] == "imagenet" || base_tokens[0] == "wikipedia" 
                || base_tokens[0] == "datacomp-image" || base_tokens[0] == "datacomp-text")
            {
                if (base_tokens[0].substr(0, 8) == "datacomp") {
                    if (base_tokens[0] == "datacomp-text") {
                        base_path += "t.";
                    } else {
                        base_path += "i.";
                    }
                }
                base_path += "norm.fvecs";
            } else {
                base_path += "fvecs";
            }
            std::cout << "[Dataloader] base_path: " << base_path << std::endl;


            auto query_tokens = split(query_name, '.');     // datacomp-text.learn or imagenet.query
            assert(query_tokens.size() == 2);
            query_path = query_prefix + query_tokens[0] + "/";
            if (query_tokens[1] == "learn") {
                query_path += "learn.";
            } else if (query_tokens[1] == "query") {
                query_path += "query.";
            }

            if (query_tokens[0] == "imagenet" || query_tokens[0] == "wikipedia"
                || query_tokens[0] == "datacomp-image" || query_tokens[0] == "datacomp-text") 
            {
                if (query_tokens[0].substr(0, 8) == "datacomp") {
                    if (query_tokens[0] == "datacomp-text") {
                        query_path += "t.";
                    } else {
                        query_path += "i.";
                    }
                }
                query_path += "norm.fvecs";
            } else {
                query_path += "fvecs";
            }
            std::cout << "[Dataloader] query_path: " << query_path << std::endl;


            if (base_tokens[0].substr(0, 8) == "datacomp" && query_tokens[0].substr(0, 8) == "datacomp") {
                gt_path = query_prefix + query_tokens[0] + "/" + query_tokens[1] + ".";
                if (query_tokens[0] == "datacomp-text" && base_tokens[0] == "datacomp-text") {
                    gt_path += "t2t.";
                } else if (query_tokens[0] == "datacomp-image" && base_tokens[0] == "datacomp-image") {
                    gt_path += "i2i.";
                } else if (query_tokens[0] == "datacomp-text" && base_tokens[0] == "datacomp-image") {
                    gt_path += "t2i.";
                } else if (query_tokens[0] == "datacomp-image" && base_tokens[0] == "datacomp-text") {
                    gt_path += "i2t.";
                }
                gt_path += base_tokens[1] + ".norm.gt.ivecs.cpu.1000";
            } else if (base_tokens[0] == query_tokens[0]) {
                gt_path = query_prefix + query_tokens[0] + "/" + query_tokens[1] + ".";
                if (query_tokens[0] == "imagenet" || query_tokens[0] == "wikipedia") {
                    gt_path += "norm";
                }
                gt_path += ".gt.ivecs.cpu.1000";
            }
            std::cout << "[Dataloader] gt_path: " << gt_path << std::endl;


            std::tie(num_base, dim_base) = utils::LoadFromFile(base_data, base_path);
            std::tie(num_query, dim_query) = utils::LoadFromFile(query_data, query_path);
            std::cout << "[Dataloader] base_data: " << num_base << " x " << dim_base << std::endl;
            std::cout << "[Dataloader] query_data: " << num_query << " x " << dim_query << std::endl;

            if (is_exists(gt_path)) {
                std::tie(num_gt, dim_gt) = utils::LoadFromFile(gt_data, gt_path);
                assert(num_gt * dim_gt == num_query * 1000);
                num_gt = num_query;
                dim_gt = 1000;
                std::cout << "[Dataloader] gt_data: " << num_gt << " x " << dim_gt << std::endl;
            } else {
                std::cout << "[Dataloader] gt not found, please check the path" << std::endl;
            }

            int metric = 0;
            if (base_tokens[0] == "imagenet" || base_tokens[0] == "wikipedia"
                || base_tokens[0] == "datacomp-image" || base_tokens[0] == "datacomp-text") {
                metric = 0;
            } else {
                metric = 1;
            }

            BaseQueryGtConfig config = {
                base_path, query_path, gt_path,
                num_base, num_query, num_gt, 
                dim_base, dim_query, dim_gt, 
                metric
            };
            return {base_data, query_data, gt_data, config};
        }

    private:
        std::string base_prefix = "/home/zhengweiguo/liuchengjun/anns/dataset/";
        std::string query_prefix = "/home/zhengweiguo/liuchengjun/anns/query/";
    };
} // namespace utils