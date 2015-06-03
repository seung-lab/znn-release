#include "network/parallel/network.hpp"

using namespace znn::v4;

int main(int argc, char** argv)
{
    int64_t x = 9;
    int64_t y = 9;
    int64_t z = 9;

    int64_t fx = 2;
    int64_t fy = 2;
    int64_t fz = 2;


    if ( argc >= 4 )
    {
        x = atoi(argv[1]);
        y = atoi(argv[2]);
        z = atoi(argv[3]);
    }

    if ( argc >= 7 )
    {
        fx = atoi(argv[4]);
        fy = atoi(argv[5]);
        fz = atoi(argv[6]);
    }

    size_t tc = 10;

    if ( argc == 8 )
    {
        tc = atoi(argv[7]);
    }

    auto v = get_cube<real>(vec3i(x,y,z));
    auto f = get_cube<real>(vec3i(fx,fy,fz));

    for ( int i = 0; i < v->num_elements(); ++i )
        v->data()[i] = i;

    for ( int i = 0; i < f->num_elements(); ++i )
        f->data()[i] = 0.001;


    v = convolve(v,f);
    v = convolve_inverse(v,f);

    zi::wall_timer wt;
    wt.reset();

    for ( size_t i = 0; i < tc; ++i )
    {
        v = convolve(v,f);
        v = convolve_inverse(v,f);
    }

    std::cout << "Elapsed: " << wt.elapsed<double>() << std::endl;
    std::cout << "Sum: " << sum(*v) << std::endl;

}
