#pragma once

#include <cudnn.h>
#include <sstream>
#include <iostream>
#include <cstddef>
#include <cstdio>

#define STRINGIFY_(s)   #s
#define STRINGIFY(s)    STRINGIFY_(s)
#if defined(__GNUC__)
#define COMPILER_NAME "GCC"
#define COMPILER_VER                                                    \
    STRINGIFY(__GNUC__) "."                                             \
    STRINGIFY(__GNUC_MINOR__) "."                                       \
    STRINGIFY(__GNUC_PATCHLEVEL__)
#elif defined(__clang_major__)
#define COMPILER_NAME "CLANG"
#define COMPILER_VER                                                    \
    STRINGIFY(__clang_major__ ) "."                                     \
    STRINGIFY(__clang_minor__) "."                                      \
    STRINGIFY(__clang_patchlevel__)
#elif defined(__INTEL_COMPILER)
#define COMPILER_NAME "ICC"
#define COMPILER_VER                                                    \
    STRINGIFY(__INTEL_COMPILER) "."                                     \
    STRINGIFY(__INTEL_COMPILER_BUILD_DATE)
#else
#define COMPILER_NAME "unknown"
#define COMPILER_VER  "???"
#endif

#define CUDNN_VERSION_STR                                               \
    STRINGIFY(CUDNN_MAJOR) "."                                          \
    STRINGIFY (CUDNN_MINOR) "."                                         \
    STRINGIFY(CUDNN_PATCHLEVEL)

#define FATAL_ERROR(s)                                                  \
    {                                                                   \
        std::cerr << __FILE__ << ':' << __LINE__ << '\n';               \
        std::cerr << std::string(s);                                    \
        std::cerr << "\nAborting...\n";                                 \
        cudaDeviceReset();                                              \
        exit(EXIT_FAILURE);                                             \
    }

#define checkCUDNN(status)                                              \
    {                                                                   \
        std::stringstream _error;                                       \
        if (status != CUDNN_STATUS_SUCCESS)                             \
        {                                                               \
            _error << "Line: " << __LINE__ << "\n";                     \
            _error << "CUDNN failure\nError: ";                         \
            _error << cudnnGetErrorString(status);                      \
            FATAL_ERROR(_error.str());                                  \
        }                                                               \
    }

#define checkCudaErrors(status)                                         \
    {                                                                   \
        std::stringstream _error;                                       \
        if (status != 0)                                                \
        {                                                               \
            _error << "Line: " << __LINE__ << "\n";                     \
            _error << "Cuda failure\nError: ";                          \
            _error << cudaGetErrorString(status);                       \
            FATAL_ERROR(_error.str());                                  \
        }                                                               \
}

#define checkCublasErrors(status)                                       \
    {                                                                   \
        std::stringstream _error;                                       \
        if (status != 0) {                                              \
            _error << "Line: " << __LINE__ << "\n";                     \
            _error << "Cublas failure\nError code " << status;          \
            FATAL_ERROR(_error.str());                                  \
        }                                                               \
    }

inline void showCudaDevices()
{
    int n_devices;
    checkCudaErrors(cudaGetDeviceCount( &n_devices ));
    std::cout << "There are " << n_devices
              << " CUDA capable devices on your machine :"
              << std::endl;

    for ( int i = 0; i < n_devices; i++)
    {
        struct cudaDeviceProp prop;
        checkCudaErrors(cudaGetDeviceProperties( &prop, i ));
        std::printf( "device %d : sms %2d  Capabilities %d.%d, "
                     "SmClock %.1f Mhz, MemSize (Mb) %d, "
                     "MemClock %.1f Mhz, Ecc=%d, boardGroupID=%d\n",
                     i, prop.multiProcessorCount, prop.major, prop.minor,
                     (float)prop.clockRate*1e-3,
                     (int)(prop.totalGlobalMem/(1024*1024)),
                     (float)prop.memoryClockRate*1e-3,
                     prop.ECCEnabled,
                     prop.multiGpuBoardGroupID);
    }
}


inline void print_descriptor( cudnnTensorDescriptor_t const & descriptor )
{
    cudnnDataType_t dataType;
    int nbDims;
    int dimA[5];
    int strideA[5];

    checkCUDNN(
        cudnnGetTensorNdDescriptor(
            descriptor,
            5, &dataType,
            &nbDims, dimA, strideA));

    std::cout << "    Type   : " << dataType << '\n';
    std::cout << "    Dims   : "
              << dimA[0] << ' ' << dimA[1] << ' '
              << dimA[2] << ' ' << dimA[3] << ' '
              << dimA[4] << '\n';

    std::cout << "    Strides: "
              << strideA[0] << ' ' << strideA[1] << ' '
              << strideA[2] << ' ' << strideA[3] << ' '
              << strideA[4] << '\n';


}


inline void print_descriptor( cudnnConvolutionDescriptor_t const & descriptor )
{
    cudnnDataType_t dataType;
    cudnnConvolutionMode_t mode;
    int nbDims;
    int pad[3];
    int stride[3];
    int upscale[3];

    checkCUDNN(
        cudnnGetConvolutionNdDescriptor(
            descriptor,
            3, &nbDims,
            pad, stride, upscale, &mode, &dataType));

    std::cout << "    Type   : " << dataType << '\n';
    std::cout << "    Mode   : " << mode << '\n';
    std::cout << "    Pad    : "
              << pad[0] << ' ' << pad[1] << ' ' << pad[2] << "\n";
    std::cout << "    Stride : "
              << stride[0] << ' ' << stride[1] << ' ' << stride[2] << "\n";
    std::cout << "    Upscale : "
              << upscale[0] << ' ' << upscale[1] << ' ' << upscale[2] << "\n";

}

inline void print_descriptor( cudnnFilterDescriptor_t const & descriptor )
{
    cudnnDataType_t dataType;
    int nbDims;
    int dims[5];

    checkCUDNN(
        cudnnGetFilterNdDescriptor(
            descriptor,
            5, &dataType,
            &nbDims, dims));

    std::cout << "    Type   : " << dataType << '\n';
    std::cout << "    Dims   : "
              << dims[0] << ' ' << dims[1] << ' '
              << dims[2] << ' ' << dims[3] << ' '
              << dims[4] << '\n';

}

inline void print_free_memory()
{
    std::size_t mf, mt;
    checkCudaErrors( cudaMemGetInfo(&mf, &mt) );

    mf /= 1024 * 1024;
    mt /= 1024 * 1024;

    std::cout << "Memory: " << mf << " / " << mt << std::endl;
}


#define DEBUG_DESCRIPTOR( descriptor )                  \
    std::cout << "Descriptor: " << #descriptor << "\n"; \
    print_descriptor(descriptor)
