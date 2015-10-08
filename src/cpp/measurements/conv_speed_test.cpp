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

    size_t rounds = 100;
    if ( argc >= 8 )
    {
        rounds = atoi(argv[7]);
    }

    size_t max_threads = 240;
    if ( argc >= 9 )
    {
        max_threads = atoi(argv[8]);
    }

    uniform_init init(0.001);
    auto v = get_cube<real>(vec3i(x,y,z));
    auto f = get_cube<real>(vec3i(fx,fy,fz));

    init.initialize(v);
    init.initialize(f);

    v = convolve(v,f);
    v = convolve_inverse(v,f);

    zi::wall_timer wt;

    for ( int i = 1; i <= max_threads; ++i )
    {
        wt.reset();

        {
            task_manager tm(i);
            for ( int j = 0; j < rounds; ++j )
            {
                tm.asap([&v,&f]()
                        {
                            auto v2 = convolve(v,f);
                            auto v3 = convolve_inverse(v2,f);
                        });
            }
        }

        std::cout << i << ' ' << wt.elapsed<double>() << '\n';
    }

    // generate
}
