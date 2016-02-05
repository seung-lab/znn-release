
#include "cpu/cpu3d.hpp"
#include "cpu_tester.hpp"

#include "cpu2/fft/optimal.hpp"
#include "cpu2/fft/base.hpp"

using namespace znn::fwd;

int main()
{

    std::cout << get_optimal_size(vec3i(123,232,111)) << std::endl;
    //optimal_radix rd;



    // {
    //     znn::fwd::network3d_descriptor fw(1);
    //      fw.conv(40,vec3i(7,7,7))
    //        .pool(vec3i(2,2,2))
    //          .conv(40,vec3i(7,7,7))
    //        .pool(vec3i(2,2,2))
    //          .conv(40,vec3i(7,7,7))
    //          .conv(40,vec3i(7,7,7))
    //          .conv(40,vec3i(7,7,7))
    //          .conv(40,vec3i(7,7,7))
    //          .conv(3,vec3i(7,7,7))
    //        .done(1,vec3i(4,4,4));

    //       fw.benchmark(2);
    // }

}
