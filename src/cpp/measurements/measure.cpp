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

    size_t min_threads = 1;

    if ( argc >= 8 )
    {
        min_threads = atoi(argv[7]);
    }

    size_t max_threads = 240;

    if ( argc >= 9 )
    {
        max_threads = atoi(argv[8]);
    }


    double speed = 1e32;

    for ( int i = min_threads; i <= max_threads; ++i )
    {
        auto res = parallel_network::network::speed_test
            (nodes, edges, {z,y,x}, i, nrnds, warmup);

        speed = std::min(speed, res.first);

        std::cout << i << ", "
                  << res.first << ", " << res.second
                  << " ( " << ( res.second * 100  / res.first ) << "% )"
                  << ";" << std::endl
                  << "____BEST: " << speed << std::endl;
    }
}
