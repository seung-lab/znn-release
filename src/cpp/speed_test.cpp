#include "network/parallel/network.hpp"
#include "network/trivial/trivial_fft_network.hpp"
#include "network/trivial/trivial_network.hpp"
#include <zi/zargs/zargs.hpp>

using namespace znn::v4;

int main(int argc, char** argv)
{
    std::vector<options> nodes, edges;

    std::string fname(argv[1]);

    parse_net_file(nodes, edges, fname);

    int64_t x = 9;
    int64_t y = 9;
    int64_t z = 9;

    if ( argc >= 5 )
    {
        x = atoi(argv[2]);
        y = atoi(argv[3]);
        z = atoi(argv[4]);
    }

    size_t tc = std::thread::hardware_concurrency();

    if ( argc == 6 )
    {
        tc = atoi(argv[5]);
    }

    std::vector<double> times(tc);

    for ( size_t i = 0; i < tc; ++i )
        times[i] =
            parallel_network::network::speed_test(nodes, edges, {z,y,x}, i+1 , 10);

    for ( size_t i = 0; i < tc; ++i )
    {
        std::cout << (i+1) << ", " << times[i] << ";\n";
    }
}
