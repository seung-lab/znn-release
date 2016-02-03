namespace znn { namespace gpu3d {

class gpu_layer
{
public:
    virtual ~gpu_layer() {}

    virtual void forward( cudnnHandle_t &, float *, float * ) = 0;
    virtual int in_memory() const = 0;
    virtual int out_memory() const = 0;
};

class pooling_layer: public gpu_layer
{
private:
    std::vector<cudnnTensorDescriptor_t> in_descs;
    std::vector<cudnnTensorDescriptor_t> out_descs;

    cudnnPoolingDescriptor_t pooling_desc;

    int in_memory_;
    int out_memory_;

    int wi_;
    int wo_;
    int di_;

    float alpha = 1;
    float beta  = 0;

public:

    void forward( cudnnHandle_t & cudnnHandle,
                  float * in,
                  float * out ) override
    {
        int dd = std::min(di_,wo_);

        for ( int x = 0, i = 0; x < dd; ++x )
            for ( int y = 0; y < dd; ++y )
                for ( int z = 0; z < dd; ++z, ++i )
                {
                    checkCUDNN( cudnnPoolingForward(cudnnHandle,
                                                    pooling_desc,
                                                    &alpha,
                                                    in_descs[i],
                                                    in + x * wi_ * wi_ + y * wi_ + z,
                                                    &beta,
                                                    out_descs[i],
                                                    out + x * wo_ * wo_ + y * wo_ + z));
                }
    }

    int in_memory() const override
    {
        return in_memory_;
    }

    int out_memory() const override
    {
        return out_memory_;
    }


    ~pooling_layer()
    {
        checkCUDNN( cudnnDestroyPoolingDescriptor(pooling_desc) );
        for ( auto & x: in_descs )
        {
            checkCUDNN( cudnnDestroyTensorDescriptor(x) );
        }
        for ( auto & x: out_descs )
        {
            checkCUDNN( cudnnDestroyTensorDescriptor(x) );
        }
    }

    pooling_layer( cudnnHandle_t cudnnHandle,
                   int n, int c, int wi, int sz, int di )
        : in_descs(di*di*di)
        , out_descs(di*di*di)
    {
        checkCUDNN( cudnnCreatePoolingDescriptor(&pooling_desc) );

        {
            const int poolDims = 3;
            int windowDimA[poolDims] = {sz,sz,sz};
            int paddingA[poolDims] = {0,0,0};
            int strideA[poolDims] = {1,1,1};
            checkCUDNN( cudnnSetPoolingNdDescriptor(pooling_desc,
                                                    CUDNN_POOLING_MAX,
                                                    poolDims,
                                                    windowDimA,
                                                    paddingA,
                                                    strideA ) );
        }


        int wo = wi - (sz - 1) * di;

        wi_ = wi;
        wo_ = wo;
        di_ = di;

        in_memory_  = n * c * wi * wi * wi * sizeof(float);
        out_memory_ = n * c * wo * wo * wo * sizeof(float);

        int dd = std::min(di_,wo_);

        for ( int x = 0, i = 0; x < dd; ++x )
            for ( int y = 0; y < dd; ++y )
                for ( int z = 0; z < dd; ++z, ++i )
                {
                    auto & in_desc  = in_descs[i];
                    auto & out_desc = out_descs[i];

                    checkCUDNN( cudnnCreateTensorDescriptor(&in_desc) );
                    checkCUDNN( cudnnCreateTensorDescriptor(&out_desc) );

                    int wx = (wi-x-1)/di+1;
                    int wy = (wi-y-1)/di+1;
                    int wz = (wi-z-1)/di+1;

                    const int nDims = 5;
                    int dimA[nDims] = {n,c,wx,wy,wz};

                    int strideA[nDims] = {c*wi*wi*wi,
                                          wi*wi*wi,
                                          wi*wi*di,
                                          wi*di,
                                          di};

                    checkCUDNN( cudnnSetTensorNdDescriptor(in_desc,
                                                           CUDNN_DATA_FLOAT,
                                                           5,
                                                           dimA,
                                                           strideA ) );


                    int dimB[nDims] = {n,c,wx+1-sz,wy+1-sz,wz+1-sz};

                    int strideB[nDims] = {c*wo*wo*wo,
                                          wo*wo*wo,
                                          wo*wo*di,
                                          wo*di,
                                          di};

                    checkCUDNN( cudnnSetTensorNdDescriptor(out_desc,
                                                           CUDNN_DATA_FLOAT,
                                                           5,
                                                           dimB,
                                                           strideB ) );

                }

    }
};


class conv_layer: public gpu_layer
{
private:
    std::vector<cudnnTensorDescriptor_t> in_descs;
    std::vector<cudnnTensorDescriptor_t> out_descs;

    cudnnTensorDescriptor_t      full_desc;
    cudnnTensorDescriptor_t      bias_desc;

    cudnnFilterDescriptor_t      filter_desc;
    cudnnConvolutionDescriptor_t conv_desc  ;

    int in_memory_;
    int out_memory_;

    int wi_;
    int wo_;
    int di_;

    float alpha = 1;
    float beta  = 0;

    float * data_;
    float * bias_data_;

    size_t workspace_size = 0;

public:

    float* data()
    {
        return data_;
    }

    float* bias_data()
    {
        return bias_data_;
    }

    void forward( cudnnHandle_t & cudnnHandle,
                  float * in,
                  float * out ) override
    {
        int dd = std::min(di_,wo_);

        void * workspace;


        if ( workspace_size )
        {
            checkCudaErrors( cudaMalloc(&workspace, workspace_size ));
        }

        std::cout << "Workspace: " << ( workspace_size / 1024 / 1024 ) << " MB\n";


        for ( int x = 0, i = 0; x < dd; ++x )
            for ( int y = 0; y < dd; ++y )
                for ( int z = 0; z < dd; ++z, ++i )
                {
                    checkCUDNN( cudnnConvolutionForward(cudnnHandle,
                                                        &alpha,
                                                        in_descs[i],
                                                        in + x * wi_ * wi_ + y * wi_ + z,
                                                        filter_desc,
                                                        data_,
                                                        conv_desc,
                                                        ( di_ == 1 ) ?
                                                        CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM
                                                        : CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM ,
                                                        workspace,
                                                        workspace_size,
                                                        &beta,
                                                        out_descs[i],
                                                        out + x * wo_ * wo_ + y * wo_ + z));
                }

        if ( workspace_size )
        {
            checkCudaErrors( cudaFree(workspace) );
        }

        float beta2 = 1;

        checkCUDNN( cudnnAddTensor( cudnnHandle,
                                    &alpha,
                                    bias_desc,
                                    bias_data_,
                                    &beta2,
                                    full_desc,
                                    out) );

        checkCUDNN( cudnnActivationForward(cudnnHandle,
                                           CUDNN_ACTIVATION_RELU,
                                           &alpha,
                                           full_desc,
                                           out,
                                           &beta,
                                           full_desc,
                                           out) );


    }

    int in_memory() const override
    {
        return in_memory_;
    }

    int out_memory() const override
    {
        return out_memory_;
    }


    ~conv_layer()
    {
        checkCudaErrors( cudaFree(data_) );
        checkCudaErrors( cudaFree(bias_data_) );

        checkCUDNN( cudnnDestroyTensorDescriptor(full_desc) );
        checkCUDNN( cudnnDestroyTensorDescriptor(bias_desc) );
        checkCUDNN( cudnnDestroyFilterDescriptor(filter_desc) );
        checkCUDNN( cudnnDestroyConvolutionDescriptor(conv_desc) );
        for ( auto & x: in_descs )
        {
            checkCUDNN( cudnnDestroyTensorDescriptor(x) );
        }
        for ( auto & x: out_descs )
        {
            checkCUDNN( cudnnDestroyTensorDescriptor(x) );
        }
    }

    conv_layer( cudnnHandle_t cudnnHandle,
                int n, int ci, int co, int wi, int sz, int di )
        : in_descs(di*di*di)
        , out_descs(di*di*di)
    {
        checkCudaErrors( cudaMalloc(&data_, sz*sz*sz*ci*co*sizeof(float) ));
        checkCudaErrors( cudaMalloc(&bias_data_, co*sizeof(float) ));

        checkCUDNN( cudnnCreateTensorDescriptor(&full_desc) );
        checkCUDNN( cudnnCreateTensorDescriptor(&bias_desc) );
        checkCUDNN( cudnnCreateFilterDescriptor(&filter_desc) );
        checkCUDNN( cudnnCreateConvolutionDescriptor(&conv_desc) );

        int wo = wi - (sz - 1) * di;

        {
            const int Dims = 5;
            int DimA[Dims] = {n,co,wo,wo,wo};
            int strideA[Dims] = {co*wo*wo*wo,
                                 wo*wo*wo,
                                 wo*wo,
                                 wo,
                                 1};

            checkCUDNN( cudnnSetTensorNdDescriptor(full_desc,
                                                   CUDNN_DATA_FLOAT,
                                                   5,
                                                   DimA,
                                                   strideA) );
        }

        {
            const int Dims = 5;
            int DimA[Dims] = {1,co,1,1,1};
            int strideA[Dims] = {co,1,1,1,1};

            checkCUDNN( cudnnSetTensorNdDescriptor(bias_desc,
                                                   CUDNN_DATA_FLOAT,
                                                   5,
                                                   DimA,
                                                   strideA) );
        }

        {
            const int Dims = 5;
            int DimA[Dims] = {co,ci,sz,sz,sz};

            checkCUDNN( cudnnSetFilterNdDescriptor(filter_desc,
                                                   CUDNN_DATA_FLOAT,
                                                   5,
                                                   DimA) );
        }

        {
            const int convDims = 3;
            int padA[convDims] = {0,0,0};
            int filterStrideA[convDims] = {1,1,1};
            int upscaleA[convDims] = {1,1,1};

            checkCUDNN( cudnnSetConvolutionNdDescriptor(conv_desc,
                                                        convDims,
                                                        padA,
                                                        filterStrideA,
                                                        upscaleA,
                                                        //CUDNN_CROSS_CORRELATION,
                                                        CUDNN_CONVOLUTION,
                                                        CUDNN_DATA_FLOAT) );
        }


        wi_ = wi;
        wo_ = wo;
        di_ = di;

        in_memory_  = n * ci * wi * wi * wi * sizeof(float);
        out_memory_ = n * co * wo * wo * wo * sizeof(float);

        int dd = std::min(di,wo_);

        for ( int x = 0, i = 0; x < dd; ++x )
            for ( int y = 0; y < dd; ++y )
                for ( int z = 0; z < dd; ++z, ++i )
                {
                    auto & in_desc  = in_descs[i];
                    auto & out_desc = out_descs[i];

                    checkCUDNN( cudnnCreateTensorDescriptor(&in_desc) );
                    checkCUDNN( cudnnCreateTensorDescriptor(&out_desc) );

                    int wx = (wi-x-1)/di+1;
                    int wy = (wi-y-1)/di+1;
                    int wz = (wi-z-1)/di+1;

                    const int nDims = 5;
                    int dimA[nDims] = {n,ci,wx,wy,wz};

                    int strideA[nDims] = {ci*wi*wi*wi,
                                          wi*wi*wi,
                                          wi*wi*di,
                                          wi*di,
                                          di};

                    checkCUDNN( cudnnSetTensorNdDescriptor(in_desc,
                                                           CUDNN_DATA_FLOAT,
                                                           5,
                                                           dimA,
                                                           strideA ) );


                    int dimB[nDims] = {n,co,wx+1-sz,wy+1-sz,wz+1-sz};

                    int dq = di;
                    if ( wo == 0 ) dq = 1;

                    int strideB[nDims] = {co*wo*wo*wo,
                                          wo*wo*wo,
                                          wo*wo*dq,
                                          wo*dq,
                                          dq};

                    checkCUDNN( cudnnSetTensorNdDescriptor(out_desc,
                                                           CUDNN_DATA_FLOAT,
                                                           5,
                                                           dimB,
                                                           strideB ) );

                    if ( di == 1 )
                    {
                        size_t what_size;

                        checkCUDNN( cudnnGetConvolutionForwardWorkspaceSize(cudnnHandle,
                                                                            in_desc,
                                                                            filter_desc,
                                                                            conv_desc,
                                                                            out_desc,
                                                                            CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM,
                                                                            &what_size));
                        workspace_size = std::max(workspace_size, what_size);
                    }


                }
    }
};


class fwd_network
{
private:
    struct xxx
    {
        int ltype;
        int num_units;
        int filter_size;
        int dilation;
    };

    std::list<xxx>          descs ;
    std::list<gpu_layer*>   layers;

    int num_input_units;
    int curr_num_units;
    int curr_dil = 1;
    int outsz;
    int insz;
    int batch;

    cudnnHandle_t cudnn_handle;

public:

    ~fwd_network()
    {
        checkCUDNN( cudnnDestroy(cudnn_handle) );
        for ( auto & a: layers )
            delete a;
    }

    fwd_network( int n )
        : num_input_units(n)
        , curr_num_units(n)
    {
        checkCUDNN( cudnnCreate(&cudnn_handle) );
    }

    fwd_network & conv( int c, int w )
    {
        descs.push_back({1,c,w,curr_dil});
        curr_num_units = c;
        return *this;
    }

    fwd_network & pool( int w )
    {
        descs.push_back({1,curr_num_units,w,curr_dil});
        curr_dil *= w;
        return *this;
    }

    void done( int n, int sz )
    {
        batch = n;
        outsz = sz;
        insz = outsz;

        uniform_init ui(0.1);

        std::reverse(descs.begin(), descs.end());
        for ( auto & a: descs )
        {
            insz = insz + ( a.filter_size - 1 ) * a.dilation;
        }

        std::reverse(descs.begin(), descs.end());
        curr_num_units = num_input_units;

        int sx = insz;

        for ( auto & a: descs )
        {
            if ( a.ltype == 1 )
            {
                conv_layer* cl = new conv_layer(cudnn_handle, batch, curr_num_units,
                                                a.num_units, sx,
                                                a.filter_size, a.dilation);


                int floats = a.filter_size*a.filter_size*a.filter_size*curr_num_units*a.num_units;
                float* dt = new float[floats];
                ui.initialize(dt, floats);

                checkCudaErrors( cudaMemcpy(cl->data(), dt,
                                            floats * sizeof(float),
                                            cudaMemcpyHostToDevice) );

                delete[] dt;

                floats = a.num_units;
                dt = new float[floats];
                ui.initialize(dt, floats);

                checkCudaErrors( cudaMemcpy(cl->bias_data(), dt,
                                            floats * sizeof(float),
                                            cudaMemcpyHostToDevice) );

                delete[] dt;

                layers.push_back(cl);

                std::cout << "Conv: " << batch << ' '
                          << curr_num_units << ' '
                          << a.num_units << ' '
                          << sx << ' '
                          << a.filter_size << ' '
                          << a.dilation << '\n';

                sx -= ( a.filter_size - 1 ) * a.dilation;
                curr_num_units = a.num_units;
                std::cout << "   out: " << sx << "\n";
            }
            else if ( a.ltype == 2 )
            {
                pooling_layer* pl = new pooling_layer(cudnn_handle, batch, curr_num_units,
                                                      sx, a.filter_size, a.dilation);

                layers.push_back(pl);

                std::cout << "Pool: " << batch << ' '
                          << curr_num_units << ' '
                          << sx << ' '
                          << a.filter_size << ' '
                          << a.dilation << '\n';

                sx -= ( a.filter_size - 1 ) * a.dilation;
                std::cout << "   out: " << sx << "\n";
            }
        }

    }

    void benchmark( int rounds )
    {
        uniform_init ui(0.1);

        int host_data_len = layers.front()->in_memory()/sizeof(float);
        int host_out_len  = outsz * outsz * outsz * curr_num_units * batch;

        float* host_data_in = new float[host_data_len];
        float* host_data_out = new float[host_out_len];

        zi::wall_timer wt;

        for ( ; rounds > 0; --rounds )
        {

            ui.initialize(host_data_in, host_data_len);

            wt.reset();

            float* input;
            float* output;

            checkCudaErrors( cudaMalloc(&input, layers.front()->in_memory()) );

            checkCudaErrors( cudaMemcpy(input, host_data_in,
                                        layers.front()->in_memory(),
                                        cudaMemcpyHostToDevice) );

            gpu_layer* last;

            for ( auto & a: layers )
            {
                last = a;
                checkCudaErrors( cudaMalloc(&output, a->out_memory()) );

                a->forward(cudnn_handle, input, output);

                //print_free_memory();

                checkCudaErrors( cudaFree(input) );
                input = output;
            }

            checkCudaErrors( cudaMemcpy(host_data_out, input,
                                        last->out_memory(),
                                        cudaMemcpyDeviceToHost) );
            checkCudaErrors( cudaFree(input) );

            double tm = wt.elapsed<double>();
            std::cout << tm << "\n";
            std::cout << static_cast<double>(outsz*outsz*outsz*batch) / tm << "\n";

        }


        delete [] host_data_in;

    }

};


}} // namespace znn::gpu
