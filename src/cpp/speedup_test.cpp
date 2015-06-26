#include "network/parallel/network.hpp"
#include "network/trivial/trivial_fft_network.hpp"
#include "network/trivial/trivial_network.hpp"
#include <zi/zargs/zargs.hpp>

using namespace znn::v4;

int main(int argc, char** argv)
{
    std::vector<options> nodes, edges;

    int64_t W = atoi(argv[1]);
    int64_t D = atoi(argv[2]);

    nodes.resize(2*D);
    edges.resize(2*D-1);

    nodes[0].push("name", "input").push("type", "input").push("size", 1);

    for ( int64_t i = 0; i < D; ++i )
    {
        edges[i*2].push("name", 2*i).push("type", "conv").push("init", "uniform")
            .push("size", "5,5,5").push("stride", "1,1,1")
            .push("input", 2*i-1).push("output",2*i);

        nodes[i*2+1].push("name",2*i).push("type","transfer")
            .push("function","rectify_linear").push("size",W);

        if ( i != D-1 )
        {
            edges[i*2+1].push("name", i*2+1).push("type", "max_filter")
                .push("size", "2,2,2").push("stride", "2,2,2")
                .push("input", 2*i).push("output",2*i+1);

            nodes[i*2+2].push("name",2*i+1).push("type","sum").push("size",W);
        }
    }

    edges[0].push("input","input");
    edges[2*D-2].push("output","output");
    nodes[2*D-1].push("name","output").push("size",1);

    int64_t x = 9;
    int64_t y = 9;
    int64_t z = 9;

    if ( argc >= 6 )
    {
        x = atoi(argv[3]);
        y = atoi(argv[4]);
        z = atoi(argv[5]);
    }

    size_t tc = std::thread::hardware_concurrency();

    if ( argc == 7 )
    {
        tc = atoi(argv[6]);
    }

    double serial =
        parallel_network::network::speed_test(nodes, edges, {z,y,x}, 1, 2);

    double parallel =
        parallel_network::network::speed_test(nodes, edges, {z,y,x}, tc, 10);

    std::cout << serial << ' ' << parallel << std::endl;
    std::cout << (serial/parallel) << std::endl;
}
