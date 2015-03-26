#include "network/trivial/trivial_network.hpp"

using namespace znn::v4;

int main()
{

    std::vector<options> nodes(9);
    std::vector<options> edges(8);

    nodes[0].push("name", "input")
        .push("type", "input")
        .push("size", 1);

    edges[0].push("name", "conv1")
        .push("type", "conv")
        .push("init", "uniform")
        .push("size", "3,3,1")
        .push("input", "input")
        .push("output", "nl1");

    nodes[1].push("name", "nl1")
        .push("type", "transfer")
        .push("function", "tanh")
        .push("function_args", "1.7159,0.6666")
        .push("size", 48);

    edges[1].push("name", "pool1")
        .push("type", "max_filter")
        .push("size", "3,3,1")
        .push("stride", "3,3,1")
        .push("input", "nl1")
        .push("output", "mp1");

    nodes[2].push("name", "mp1")
        .push("type", "sum")
        .push("size", 48);

    edges[2].push("name", "conv2")
        .push("type", "conv")
        .push("init", "uniform")
        .push("size", "4,4,1")
        .push("input", "mp1")
        .push("output", "nl2");

    nodes[3].push("name", "nl2")
        .push("type", "transfer")
        .push("function", "tanh")
        .push("function_args", "1.7159,0.6666")
        .push("size", 48);

    edges[3].push("name", "pool2")
        .push("type", "max_filter")
        .push("size", "2,2,1")
        .push("stride", "2,2,1")
        .push("input", "nl2")
        .push("output", "mp2");

    nodes[4].push("name", "mp2")
        .push("type", "sum")
        .push("size", 48);

    edges[4].push("name", "conv3")
        .push("type", "conv")
        .push("init", "uniform")
        .push("size", "4,4,1")
        .push("input", "mp2")
        .push("output", "nl3");

    nodes[5].push("name", "nl3")
        .push("type", "transfer")
        .push("function", "tanh")
        .push("function_args", "1.7159,0.6666")
        .push("size", 48);

    edges[5].push("name", "pool3")
        .push("type", "max_filter")
        .push("size", "2,2,1")
        .push("stride", "2,2,1")
        .push("input", "nl3")
        .push("output", "mp3");

    nodes[6].push("name", "mp3")
        .push("type", "sum")
        .push("size", 48);

    edges[6].push("name", "conv4")
        .push("type", "conv")
        .push("init", "uniform")
        .push("size", "3,3,1")
        .push("input", "mp3")
        .push("output", "nl4");

    nodes[7].push("name", "nl4")
        .push("type", "transfer")
        .push("function", "tanh")
        .push("function_args", "1.7159,0.6666")
        .push("size", 100);

    edges[7].push("name", "conv5")
        .push("type", "conv")
        .push("init", "uniform")
        .push("size", "1,1,1")
        .push("input", "nl4")
        .push("output", "output");

    nodes[8].push("name", "output")
        .push("type", "transfer")
        .push("function", "linear")
        .push("function_args", "1,0")
        .push("size", 2);



    trivial_network tn(nodes, edges, {1,1,1});
    tn.set_eta(0.1);

    std::map<std::string, std::vector<cube_p<double>>> in;
    in["input"].push_back(get_cube<double>({65,65,3}));

    auto r = tn.forward(std::move(in));

    for ( auto & it: r )
    {
        std::cout << it.first << "\n";
        for ( auto & v: it.second )
        {
            std::cout << size(*v) << "\n\n";
        }
    }

    r = tn.backward(std::move(r));

    for ( auto & it: r )
    {
        std::cout << it.first << "\n";
        for ( auto & v: it.second )
        {
            std::cout << size(*v) << "\n\n";
        }
    }

    auto state = tn.serialize();

    std::cout << "Node Groups: " << std::endl;
    for ( auto & n: state.first )
    {
        n.dump();
        std::cout << "\n";
    }

    std::cout << "Edge Groups: " << std::endl;
    for ( auto & n: state.second )
    {
        n.dump();
        std::cout << "\n";
    }

    return 0;
}
