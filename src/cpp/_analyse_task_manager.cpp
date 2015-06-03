#define ZNN_ANALYSE_TASK_MANAGER 1

#include "network/parallel/network.hpp"
#include <zi/zargs/zargs.hpp>

using namespace znn::v4;

int main(int argc, char** argv)
{
    std::vector<options> nodes(7);
    std::vector<options> edges(6);

    nodes[0].push("name", "input")
        .push("type", "input")
        .push("size", 1);

    edges[0].push("name", "conv1")
        .push("type", "conv")
        .push("init", "uniform")
        .push("size", "7,7,7")
        .push("stride", "1,1,1")
        .push("input", "input")
        .push("output", "nl1");

    nodes[1].push("name", "nl1")
        .push("type", "transfer")
        .push("function", "rectify_linear")
        .push("size", 16);

    edges[1].push("name", "conv2")
        .push("type", "conv")
        .push("init", "uniform")
        .push("size", "7,7,7")
        .push("stride", "1,1,1")
        .push("input", "nl1")
        .push("output", "nl2");

    nodes[2].push("name", "nl2")
        .push("type", "transfer")
        .push("function", "rectify_linear")
        .push("size", 16);

    edges[2].push("name", "conv3")
        .push("type", "conv")
        .push("init", "uniform")
        .push("size", "7,7,7")
        .push("stride", "1,1,1")
        .push("input", "nl2")
        .push("output", "nl3");

    nodes[3].push("name", "nl3")
        .push("type", "transfer")
        .push("function", "rectify_linear")
        .push("size", 16);

    edges[3].push("name", "conv4")
        .push("type", "conv")
        .push("init", "uniform")
        .push("size", "7,7,7")
        .push("stride", "1,1,1")
        .push("input", "nl3")
        .push("output", "nl4");

    nodes[4].push("name", "nl4")
        .push("type", "transfer")
        .push("function", "rectify_linear")
        .push("size", 16);

    edges[4].push("name", "conv5")
        .push("type", "conv")
        .push("init", "uniform")
        .push("size", "7,7,7")
        .push("stride", "1,1,1")
        .push("input", "nl4")
        .push("output", "nl5");

    nodes[5].push("name", "nl5")
        .push("type", "transfer")
        .push("function", "rectify_linear")
        .push("size", 16);

    edges[5].push("name", "conv6")
        .push("type", "conv")
        .push("init", "uniform")
        .push("size", "1,1,1")
        .push("stride", "1,1,1")
        .push("input", "nl5")
        .push("output", "nl6");

    nodes[6].push("name", "nl6")
        .push("type", "transfer")
        .push("function", "rectify_linear")
        .push("size", 1);


    int64_t x = 9;
    int64_t y = 9;
    int64_t z = 9;

    if ( argc >= 4 )
    {
        x = atoi(argv[1]);
        y = atoi(argv[2]);
        z = atoi(argv[3]);
    }

    size_t tc = std::thread::hardware_concurrency();

    if ( argc == 5 )
    {
        tc = atoi(argv[4]);
    }

    parallel_network::network::optimize(nodes, edges, {z,y,x}, tc , 10);

}
