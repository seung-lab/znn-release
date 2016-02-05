
#include "cpu/cpu3d.hpp"
#include "cpu_tester.hpp"

using namespace znn::fwd;

int main()
{

    {
        znn::fwd::network3d_descriptor fw(1);
         fw.conv(40,vec3i(7,7,7))
           .pool(vec3i(2,2,2))
             .conv(40,vec3i(7,7,7))
           .pool(vec3i(2,2,2))
             .conv(40,vec3i(7,7,7))
             .conv(40,vec3i(7,7,7))
             .conv(40,vec3i(7,7,7))
             .conv(40,vec3i(7,7,7))
             .conv(3,vec3i(7,7,7))
           .done(1,vec3i(4,4,4));

          fw.benchmark(2);
    }

}
