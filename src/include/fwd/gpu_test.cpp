
#include "gpu/gpu2d.hpp"
#include "gpu/gpu3d.hpp"
#include "descriptor.hpp"

using znn::fwd::vec2i;
using znn::fwd::vec3i;

int main()
{
    int version = (int)cudnnGetVersion();
    printf("cudnnGetVersion() : %d , CUDNN_VERSION from cudnn.h : %d (%s)\n", version, CUDNN_VERSION, CUDNN_VERSION_STR);
    printf("Host compiler version : %s %s\r", COMPILER_NAME, COMPILER_VER);
    showCudaDevices();

    int device = 0;
    checkCudaErrors( cudaSetDevice(device) );
    std::cout << "Using device " << device << std::endl;

    struct cudaDeviceProp prop;
    checkCudaErrors(cudaGetDeviceProperties( &prop, device ));
    double globalMem = prop.totalGlobalMem/double(1024*1024);

    std::cout << "Memory: " << globalMem << std::endl;



    // {
    //     znn::fwd::gpu2d::fwd_network fw(1);
    //     fw.conv(12,vec2i(6,6))
    //         .pool(vec2i(2,2))
    //         .conv(12,vec2i(6,6))
    //         .pool(vec2i(2,2))
    //         .conv(24,vec2i(6,6))
    //         .pool(vec2i(2,2))
    //         .conv(24,vec2i(6,6))
    //         .pool(vec2i(2,2))
    //         .conv(48,vec2i(5,5))
    //         .conv(4,vec2i(1,1))
    //          .done(1,vec2i(1024,1024));

    //      fw.benchmark(2);
    // }


    // {
    //     znn::fwd::gpu3d::fwd_network fw(1);
    //     fw.conv(12,vec3i(6,6,1))
    //         .pool(vec3i(2,2,1))
    //         .conv(24,vec3i(4,4,1))
    //         .pool(vec3i(2,2,1))
    //         .conv(36,vec3i(4,4,4))
    //         .pool(vec3i(2,2,2))
    //         .conv(48,vec3i(4,4,2))
    //         .pool(vec3i(2,2,2))
    //         .conv(48,vec3i(4,4,2))
    //         .conv(4,vec3i(1,1,1))
    //         .done(1,vec3i(256,512,32));

    //      fw.benchmark(2);
    // }



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
           .done(1,vec3i(64,64,64));

          fw.benchmark(2);
    }
    //     {
    //     znn::fwd::network3d_descriptor fw(1);
    //      fw.conv(10,vec3i(7,7,7))
    //          .conv(10,vec3i(7,7,7))
    //          .conv(10,vec3i(7,7,7))
    //          .conv(10,vec3i(7,7,7))
    //          .conv(10,vec3i(7,7,7))
    //          .conv(1,vec3i(1,1,1))
    //          .done(1,vec3i(128,128,128));

    //       fw.benchmark(2);
    // }

    // znn::fwd::network3d_descriptor fw(1);


    // fw.conv(48,vec3i(9,9,9))
    //     .pool(vec3i(2,2,2))
    //     .conv(48,vec3i(9,9,9))
    //     .pool(vec3i(2,2,2))
    //     .conv(48,vec3i(9,9,9))
    //     .conv(48,vec3i(9,9,9))
    //     .conv(48,vec3i(9,9,9))
    //     .conv(4,vec3i(9,9,9))
    //     .done(1,vec3i(64,64,64));

    // fw.benchmark(2);

    cudaDeviceReset();

}
