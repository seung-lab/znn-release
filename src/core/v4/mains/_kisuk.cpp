#include "network/parallel/network.hpp"
#include <zi/zargs/zargs.hpp>

using namespace znn::v4;

int main(int argc, char** argv)
{
    std::vector<options> nodes(20);
    std::vector<options> edges(19);

    nodes[0].push("name", "input")
        .push("type", "input")
        .push("size", 1);

    edges[0].push("name", "conv1a")
        .push("type", "conv")
        .push("init", "uniform")
        .push("size", "1,3,3")
        .push("stride", "1,1,1")
        .push("input", "input")
        .push("output", "nl1");

    nodes[1].push("name", "nl1")
        .push("type", "transfer")
        .push("function", "rectify_linear")
        .push("size", 24);


    edges[1].push("name", "conv1b")
        .push("type", "conv")
        .push("init", "uniform")
        .push("size", "1,3,3")
        .push("stride", "1,1,1")
        .push("input", "nl1")
        .push("output", "nl2");

    nodes[2].push("name", "nl2")
        .push("type", "transfer")
        .push("function", "rectify_linear")
        .push("size", 24);


    edges[2].push("name", "conv1c")
        .push("type", "conv")
        .push("init", "uniform")
        .push("size", "1,2,2")
        .push("stride", "1,1,1")
        .push("input", "nl2")
        .push("output", "nl3");

    nodes[3].push("name", "nl3")
        .push("type", "transfer")
        .push("function", "tanh")
        .push("size", 24);

    edges[3].push("name", "pool1")
        .push("type", "max_filter")
        .push("size", "1,2,2")
        .push("stride", "1,2,2")
        .push("input", "nl3")
        .push("output", "nl4");

    nodes[4].push("name", "nl4")
        .push("type", "transfer")
        .push("function", "rectify_linear")
        .push("size", 24);

    edges[4].push("name", "conv2a")
        .push("type", "conv")
        .push("init", "uniform")
        .push("size", "1,3,3")
        .push("stride", "1,1,1")
        .push("input", "nl4")
        .push("output", "nl5");

    nodes[5].push("name", "nl5")
        .push("type", "transfer")
        .push("function", "rectify_linear")
        .push("size", 36);


    edges[5].push("name", "conv2b")
        .push("type", "conv")
        .push("init", "uniform")
        .push("size", "1,3,3")
        .push("stride", "1,1,1")
        .push("input", "nl5")
        .push("output", "nl6");

    nodes[6].push("name", "nl6")
        .push("type", "transfer")
        .push("function", "rectify_linear")
        .push("size", 36);


    edges[6].push("name", "conv2c")
        .push("type", "conv")
        .push("init", "uniform")
        .push("size", "1,3,3")
        .push("stride", "1,1,1")
        .push("input", "nl6")
        .push("output", "nl7");

    nodes[7].push("name", "nl7")
        .push("type", "transfer")
        .push("function", "tanh")
        .push("size", 36);

    edges[7].push("name", "pool2")
        .push("type", "max_filter")
        .push("size", "1,2,2")
        .push("stride", "1,1,1")
        .push("input", "nl7")
        .push("output", "nl8");

    nodes[8].push("name", "nl8")
        .push("type", "transfer")
        .push("function", "rectify_linear")
        .push("size", 36);



// 3

    edges[8].push("name", "conv3a")
        .push("type", "conv")
        .push("init", "uniform")
        .push("size", "1,3,3")
        .push("stride", "1,1,1")
        .push("input", "nl8")
        .push("output", "nl9");

    nodes[9].push("name", "nl9")
        .push("type", "transfer")
        .push("function", "rectify_linear")
        .push("size", 36);


    edges[9].push("name", "conv3b")
        .push("type", "conv")
        .push("init", "uniform")
        .push("size", "1,3,3")
        .push("stride", "1,1,1")
        .push("input", "nl9")
        .push("output", "nl10");

    nodes[10].push("name", "nl10")
        .push("type", "transfer")
        .push("function", "tanh")
        .push("size", 36);

    edges[10].push("name", "pool3")
        .push("type", "max_filter")
        .push("size", "1,2,2")
        .push("stride", "1,1,1")
        .push("input", "nl10")
        .push("output", "nl11");

    nodes[11].push("name", "nl11")
        .push("type", "transfer")
        .push("function", "rectify_linear")
        .push("size", 36);


// 4

    edges[11].push("name", "conv4a")
        .push("type", "conv")
        .push("init", "uniform")
        .push("size", "2,3,3")
        .push("stride", "1,1,1")
        .push("input", "nl11")
        .push("output", "nl12");

    nodes[12].push("name", "nl12")
        .push("type", "transfer")
        .push("function", "rectify_linear")
        .push("size", 48);


    edges[12].push("name", "conv4b")
        .push("type", "conv")
        .push("init", "uniform")
        .push("size", "2,3,3")
        .push("stride", "1,1,1")
        .push("input", "nl12")
        .push("output", "nl13");

    nodes[13].push("name", "nl13")
        .push("type", "transfer")
        .push("function", "tanh")
        .push("size", 48);

    edges[13].push("name", "pool4")
        .push("type", "max_filter")
        .push("size", "2,2,2")
        .push("stride", "1,1,1")
        .push("input", "nl13")
        .push("output", "nl14");

    nodes[14].push("name", "nl14")
        .push("type", "transfer")
        .push("function", "rectify_linear")
        .push("size", 48);


// 5

    edges[14].push("name", "conv5a")
        .push("type", "conv")
        .push("init", "uniform")
        .push("size", "2,3,3")
        .push("stride", "1,1,1")
        .push("input", "nl14")
        .push("output", "nl15");

    nodes[15].push("name", "nl15")
        .push("type", "transfer")
        .push("function", "rectify_linear")
        .push("size", 60);


    edges[15].push("name", "conv5b")
        .push("type", "conv")
        .push("init", "uniform")
        .push("size", "2,3,3")
        .push("stride", "1,1,1")
        .push("input", "nl15")
        .push("output", "nl16");

    nodes[16].push("name", "nl16")
        .push("type", "transfer")
        .push("function", "tanh")
        .push("size", 60);

    edges[16].push("name", "pool5")
        .push("type", "max_filter")
        .push("size", "2,2,2")
        .push("stride", "1,1,1")
        .push("input", "nl16")
        .push("output", "nl17");

    nodes[17].push("name", "nl17")
        .push("type", "transfer")
        .push("function", "rectify_linear")
        .push("size", 60);

 // 6

    edges[17].push("name", "conv6")
        .push("type", "conv")
        .push("init", "uniform")
        .push("size", "3,3,3")
        .push("stride", "1,1,1")
        .push("input", "nl17")
        .push("output", "nl18");

    nodes[18].push("name", "nl18")
        .push("type", "transfer")
        .push("function", "rectify_linear")
        .push("size", 200);

    edges[18].push("name", "convx")
        .push("type", "conv")
        .push("init", "uniform")
        .push("size", "1,1,1")
        .push("stride", "1,1,1")
        .push("input", "nl18")
        .push("output", "nl19");

    nodes[19].push("name", "nl19")
        .push("type", "transfer")
        .push("function", "logistics")
        .push("size", 2);


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
