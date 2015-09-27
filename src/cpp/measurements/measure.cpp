#include "network/parallel/network.hpp"
#include "network/trivial/trivial_fft_network.hpp"
#include "network/trivial/trivial_network.hpp"
#include "network/helpers.hpp"
#include <zi/zargs/zargs.hpp>
#include <fstream>

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

    size_t warmup = 3;

    if ( argc >= 6 )
    {
        warmup = atoi(argv[5]);
    }

    size_t nrnds = 31;

    if ( argc >= 7 )
    {
        nrnds = atoi(argv[6]);
    }

    size_t max_threads = 240;

    std::vector<double> speeds(max_threads+1);

    for ( int i = 18; i <= max_threads; ++i )
    {
        auto res = parallel_network::network::speed_test
            (nodes, edges, {z,y,x}, i, nrnds, warmup);

        speeds[i] = res.first;

        std::cout << i << ", "
                  << res.first << ", " << res.second
                  << " ( " << ( res.second * 100  / res.first ) << "% )"
                  << ";" << std::endl
                  << "____SPEEDUP: " << ( speeds[1] / speeds[i] ) << std::endl;
    }
}
