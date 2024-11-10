#include <string>
#include <vector>
#include <utils/dataloader.hpp>



// ./test_dataloader datacomp-image.base datacomp-text.learn.query
int main(int argc, char **argv)
{
    std::string data_name = std::string(argv[1]);
    std::string query_name = std::string(argv[2]);
    utils::DataLoader data_loader;
    auto [data, query, gt, cfg] = data_loader.load(data_name, query_name);

    return 0;
}