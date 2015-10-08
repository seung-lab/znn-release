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

    size_t rounds = 100;
    if ( argc >= 5 )
    {
        rounds = atoi(argv[4]);
    }

    size_t max_threads = 240;
    if ( argc >= 6 )
    {
        max_threads = atoi(argv[5]);
    }

    uniform_init init(1);
    auto v = get_cube<real>({x,y,z});
    init.initialize(v);

    fftw::transformer fft({x,y,z});
    zi::wall_timer wt;

    for ( int i = 1; i <= max_threads; ++i )
    {
        wt.reset();

        {
            task_manager tm(i);
            for ( int j = 0; j < rounds; ++j )
            {
                tm.asap([&v,&fft]()
                        {
                            auto v2 = get_copy(*v);
                            auto t = fft.forward(std::move(v2));
                            v2 = fft.backward(std::move(t));
                        });
            }
        }

        std::cout << i << ' ' << wt.elapsed<double>() << '\n';
    }

    // generate
}
