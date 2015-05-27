#include "network/parallel/network.hpp"
#include <zi/zargs/zargs.hpp>

using namespace znn::v4;

int main(int argc, char** argv)
{
    std::vector<options> nodes(11);
    std::vector<options> edges(10);

    nodes[0].push("name", "input")
        .push("type", "input")
        .push("size", 1);

    edges[0].push("name", "conv1")
        .push("type", "conv")
        .push("init", "uniform")
        .push("size", "1,6,6")
        .push("stride", "1,1,1")
        .push("input", "input")
        .push("output", "nl1");

    nodes[1].push("name", "nl1")
        .push("type", "transfer")
        .push("function", "rectify_linear")
        .push("size", 12);

    edges[1].push("name", "pool1")
        .push("type", "max_pool")
        .push("size", "1,2,2")
        .push("stride", "1,2,2")
        .push("input", "nl1")
        .push("output", "nl2");

    nodes[2].push("name", "nl2")
        .push("type", "transfer")
        .push("function", "rectify_linear")
        .push("size", 12);

    edges[2].push("name", "conv2")
        .push("type", "conv")
        .push("init", "uniform")
        .push("size", "1,6,6")
        .push("stride", "1,1,1")
        .push("input", "nl2")
        .push("output", "nl3");

    nodes[3].push("name", "nl3")
        .push("type", "transfer")
        .push("function", "rectify_linear")
        .push("size", 12);

    edges[3].push("name", "pool2")
        .push("type", "max_pool")
        .push("size", "1,2,2")
        .push("stride", "1,2,2")
        .push("input", "nl3")
        .push("output", "nl4");

    nodes[4].push("name", "nl4")
        .push("type", "transfer")
        .push("function", "rectify_linear")
        .push("size", 12);

    edges[4].push("name", "conv3")
        .push("type", "conv")
        .push("init", "uniform")
        .push("size", "1,6,6")
        .push("stride", "1,1,1")
        .push("input", "nl4")
        .push("output", "nl5");

    nodes[5].push("name", "nl5")
        .push("type", "transfer")
        .push("function", "rectify_linear")
        .push("size", 24);

    edges[5].push("name", "pool3")
        .push("type", "max_pool")
        .push("size", "1,2,2")
        .push("stride", "1,2,2")
        .push("input", "nl5")
        .push("output", "nl6");

    nodes[6].push("name", "nl6")
        .push("type", "transfer")
        .push("function", "rectify_linear")
        .push("size", 24);

    edges[6].push("name", "conv4")
        .push("type", "conv")
        .push("init", "uniform")
        .push("size", "1,6,6")
        .push("stride", "1,1,1")
        .push("input", "nl6")
        .push("output", "nl7");

    nodes[7].push("name", "nl7")
        .push("type", "transfer")
        .push("function", "rectify_linear")
        .push("size", 24);

    edges[7].push("name", "pool4")
        .push("type", "max_pool")
        .push("size", "1,2,2")
        .push("stride", "1,2,2")
        .push("input", "nl7")
        .push("output", "nl8");

    nodes[8].push("name", "nl8")
        .push("type", "transfer")
        .push("function", "rectify_linear")
        .push("size", 24);


    edges[8].push("name", "conv5")
        .push("type", "conv")
        .push("init", "uniform")
        .push("size", "1,5,5")
        .push("stride", "1,1,1")
        .push("input", "nl8")
        .push("output", "nl9");


    nodes[9].push("name", "nl9")
        .push("type", "transfer")
        .push("function", "rectify_linear")
        .push("size", 48);


    edges[9].push("name", "conv6")
        .push("type", "conv")
        .push("init", "uniform")
        .push("size", "1,1,1")
        .push("stride", "1,1,1")
        .push("input", "nl9")
        .push("output", "nl10");


    nodes[10].push("name", "nl10")
        .push("type", "transfer")
        .push("function", "rectify_linear")
        .push("size", 4);

    int64_t x = 10;
    int64_t y = 10;
    int64_t z = 1;

    if ( argc >= 3 )
    {
        x = atoi(argv[1]);
        y = atoi(argv[2]);
    }

    size_t tc = std::thread::hardware_concurrency();

    if ( argc == 4 )
    {
        tc = atoi(argv[3]);
    }

    parallel_network::network::optimize(nodes, edges, {z,y,x}, tc , 10);

}
