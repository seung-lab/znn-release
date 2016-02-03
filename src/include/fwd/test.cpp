#include "cpu/cpu3d.hpp"

#include <iostream>

void printx( int x )
{
    std::cout << x << "\n";
}

template<typename T>
void print_all( const T* t, size_t c)
{
    for ( size_t i = 0; i < c; ++i )
    {
        if ( i > 0 ) std::cout << ' ';
        std::cout << t[i];
    }

    std::cout << std::endl;
}


int main()
{
    using znn::fwd::vec3i;
    using znn::fwd::real;

    znn::fwd::task_package p(100000,10);

    znn::fwd::cpu3d::conv_layer cl(p, 1, 5, 12, vec3i(155,155,155), vec3i(5,5,5));

    // znn::fwd::pooling_layer pl(p, 2, 2, vec3i(11,11,11), vec3i(2,2,2));

    real* data = znn::fwd::znn_malloc<real>(155*155*155*1*5);

    real * xx = cl.forward(data);

    print_all(xx, 4000);



    //for ( int i = 0; i < 4; ++i )
    //     std::cout << xx[i] << "\n";

    std::cout << "HERE!!!!\n";
    // for ( int i = 0; i < 1000; ++i )
    //     p.add_task(printx, i);

    // p.execute(100);

    // std::cout << "DONE\n";

    // for ( int i = 0; i < 1000; ++i )
    //     p.add_task(printx, i);

    // p.execute(100);
}
