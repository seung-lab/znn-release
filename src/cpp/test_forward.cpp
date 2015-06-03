#include "network/parallel/network.hpp"
#include "network/trivial/trivial_forward_network.hpp"
#include <zi/zargs/zargs.hpp>

using namespace znn::v4;

int main(int argc, char** argv)
{
    int64_t x = 48;
    int64_t y = 48;
    int64_t z = 9;

    if ( argc >= 4 )
    {
        x = atoi(argv[1]);
        y = atoi(argv[2]);
        z = atoi(argv[3]);
    }

    size_t nx = 1; //std::thread::hardware_concurrency();

    if ( argc >= 5 )
    {
        nx = atoi(argv[4]);
    }

    size_t nt = 36;

    if ( argc >= 6 )
    {
        nt = atoi(argv[5]);
    }

    trivial_forward_network::forward_network net(nt);
    net.add_layer(vec3i(1,6,6),vec3i(1,2,2),12);
    net.add_layer(vec3i(1,4,4),vec3i(1,2,2),24);
    net.add_layer(vec3i(4,4,4),vec3i(2,2,2),36);
    net.add_layer(vec3i(2,4,4),vec3i(2,2,2),48);
    net.add_layer(vec3i(2,4,4),vec3i(1,1,1),48);
    net.add_layer(vec3i(1,1,1),vec3i(1,1,1),4);

    auto in_size = net.init( vec3i(z,y,x) );

    net.warmup();

    std::cout << "WARMUP DONE!" << std::endl;

    auto initf = std::make_shared<gaussian_init>(0,0.01);

    //size_t nx = 50;

    std::vector<std::vector<cube_p<real>>> in(1);

    for ( size_t i = 0; i < nx; ++i )
    {
        in[0].push_back(get_cube<real>(in_size));
        initf->initialize(in[0][i]);
    }

    net.forward(in);

}
