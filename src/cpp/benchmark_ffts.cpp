#include "network/parallel/network.hpp"

using namespace znn::v4;

int main(int argc, char** argv)
{
    int64_t x = 9;
    int64_t y = 9;
    int64_t z = 9;

    if ( argc >= 4 )
    {
        x = atoi(argv[1]);
        y = atoi(argv[2]);
        z = atoi(argv[3]);
    }

    size_t tc = 10;

    if ( argc == 5 )
    {
        tc = atoi(argv[4]);
    }

    auto v = get_cube<real>(vec3i(x,y,z));

    for ( int i = 0; i < v->num_elements(); ++i )
        v->data()[i] = i;

    auto f = fftw::forward(std::move(v));
    v = fftw::backward(std::move(f), vec3i(x,y,z));
    *v /= (x*y*z);

    zi::wall_timer wt;
    wt.reset();

    for ( size_t i = 0; i < tc; ++i )
    {
        f = fftw::forward(std::move(v));
        v = fftw::backward(std::move(f), vec3i(x,y,z));
        *v /= (x*y*z);
    }

    std::cout << "Elapsed: " << wt.elapsed<double>() << std::endl;
    std::cout << "Sum: " << sum(*v) << std::endl;

}
