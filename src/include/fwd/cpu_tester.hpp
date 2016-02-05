#pragma once

#include "types.hpp"
#include "assert.hpp"
//#include "gpu/gpu3d.hpp"
//#include "cpu/cpu3d.hpp"
#include "init.hpp"

namespace znn { namespace fwd {

template<class Conv>
class network3d_descriptor
{
public:
    struct layer_descriptor
    {
        int layer_type   ;
        int fin          ;
        int fout         ;
        vec3i filter_size;
        vec3i image_size = vec3i::one;
        int   batch_size = 1;

        layer_descriptor( int n, int a, int b, vec3i const & s )
            : layer_type(n)
            , fin(a)
            , fout(b)
            , filter_size(s)
        {}
    };

private:
    std::list<layer_descriptor> descriptors;

    //int num_input_units;
    int curr_num_units;

    vec3i current_dilation = vec3i::one;

    vec3i out_size_;
    vec3i in_size_;

    int batch_size_;

public:
    vec3i fov() const
    {
        return in_size_ + vec3i::one - out_size_;
    }

public:

    network3d_descriptor( int n )
        : curr_num_units(n)
    { }

    network3d_descriptor & conv( int c, vec3i const & w )
    {
        descriptors.push_front(layer_descriptor(1,curr_num_units,c,w));
        curr_num_units = c;
        return *this;
    }

    network3d_descriptor & conv( int c, int w )
    {
        return conv(c, vec3i(w,w,w) );
    }

    network3d_descriptor & pool( vec3i const & w )
    {
        descriptors.push_front
            (layer_descriptor(2,curr_num_units,curr_num_units,w));
        current_dilation *= w;
        return *this;
    }

    void done( int n, vec3i const & sz )
    {
        STRONG_ASSERT( (sz % current_dilation) == vec3i::zero );

        batch_size_ = n;
        out_size_ = sz;

        int batch_size = n * current_dilation[0] *
            current_dilation[1] * current_dilation[2];

        vec3i out_size = sz / current_dilation;

        for ( auto & d: descriptors )
        {
            if ( d.layer_type == 1 )
            {
                vec3i in_size = out_size + d.filter_size - vec3i::one;
                d.image_size = in_size;
                d.batch_size = batch_size;

                std::cout << "Conv: " << batch_size << "  \t"
                          << d.fin << '\t'
                          << d.fout << '\t'
                          << in_size << '\t'
                          << out_size << '\t'
                          << d.filter_size << '\n';

                out_size = in_size;
            }
            else if ( d.layer_type == 2 )
            {
	      vec3i in_size = out_size * d.filter_size + d.filter_size - vec3i::one;
                batch_size /=
                    d.filter_size[0] * d.filter_size[1] * d.filter_size[2];

                d.batch_size = batch_size;
                d.image_size = in_size;

                std::cout << "Pool: " << batch_size << "  \t"
                          << d.fin << '\t'
                          << d.fin << '\t'
                          << in_size << '\t'
                          << out_size << '\t'
                          << d.filter_size << '\n';

                out_size = in_size;
            }
        }

        in_size_ = out_size;

        std::cout << "Network :: FOV = "
                  << fov() << "  Input = "
                  << in_size_ << "  Output = "
                  << out_size_ << "  Batch = "
                  << batch_size_ << std::endl;
    }

    void benchmark( int rounds )
    {
        task_package  cpu_handle(1000000, 36);

        std::vector<cpu3d::cpu_layer*> cpu_layers;

        uniform_init ui(0.1);
        for ( auto & d: descriptors )
        {
            if ( d.layer_type == 1 )
            {
                Conv* cc = new Conv
                    ( cpu_handle, d.batch_size,
                      d.fin, d.fout, d.image_size, d.filter_size );

                int floats = d.fin * d.fout *
                    d.filter_size[0] * d.filter_size[1] * d.filter_size[2];

                ui.initialize(cc->kernel_data(), floats);
                ui.initialize(cc->bias_data(), d.fout);

                cpu_layers.push_back(cc);
            }
            else if ( d.layer_type == 2 )
            {
                cpu3d::pooling_layer* cp = new cpu3d::pooling_layer
                    ( cpu_handle, d.batch_size,
                      d.fout, d.image_size, d.filter_size );

                cpu_layers.push_back(cp);
            }
        }

        std::reverse( cpu_layers.begin(), cpu_layers.end() );

        zi::wall_timer wt, wtt;

        int in_data_len  = cpu_layers[0]->in_memory() / sizeof(real);

        for ( ; rounds > 0; --rounds )
        {
            wtt.reset();
            real* in_data = znn_malloc<real>(in_data_len);

            ui.initialize( in_data, in_data_len );

            wt.reset();

            for ( size_t l = 0; l < cpu_layers.size(); ++l )
            {
                std::cout << "Layer " << (l+1) << std::endl;
                wt.reset();

                // CPU first
                in_data = cpu_layers[l]->forward(in_data);

                std::cout << "  CPU took: " << wt.elapsed<double>() << std::endl;
            }

            std::cout << "TOTAL: " << wtt.elapsed<double>() << std::endl;
            wt.reset();

            znn_free(in_data);
        }


        for ( auto & a: cpu_layers ) delete a;

    }

};


}} // namespace znn::fwd
