#pragma once

#include "task_package.hpp"
#include "../types.hpp"
#include "../assert.hpp"
#include "fftw.hpp"
#include "pooling.hpp"
#include "malloc.hpp"

namespace znn { namespace fwd { namespace cpu3d {

class cpu_layer
{
public:
    virtual ~cpu_layer() {}

    virtual real * forward( real * ) = 0;
    virtual int  in_memory()  const = 0;
    virtual int  out_memory() const = 0;
};


class conv_layer: public cpu_layer
{
private:
    task_package & handle_;

    long_t n_    ;
    long_t fin_  ;
    long_t fout_ ;
    vec3i  is_   ;
    vec3i  fs_   ;

    int in_memory_ ;
    int out_memory_;

    real * kernel_data_ ;
    real * bias_data_   ;

    fft_transformer* fft_iimage_;
    fft_transformer* fft_oimage_;
    fft_transformer* fft_kernel_;

public:

    real* kernel_data()
    {
        return kernel_data_;
    }

    real* bias_data()
    {
        return bias_data_;
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
        znn_free(kernel_data_);
        znn_free(bias_data_);
    }

public:
    conv_layer( task_package& handle,
                int n, int fin, int fout,
                vec3i const & is,
                vec3i const & fs )
        : handle_(handle)
        , n_(n)
        , fin_(fin)
        , fout_(fout)
        , is_(is)
        , fs_(fs)
    {

        kernel_data_ = znn_malloc<real>(fin * fout * fs[0] * fs[1] * fs[2]);
        bias_data_   = znn_malloc<real>(fout);

        in_memory_ = n * fin * is[0] * is[1] * is[2] * sizeof(real);

        vec3i os = is + vec3i::one - fs;

        out_memory_ = fout * os[0] * os[1] * os[2] * sizeof(real);

        os[0] = is[0];

        fft_iimage_ = fft_plans.get(is,is);
        fft_oimage_ = fft_plans.get(os,is);

        vec3i x = fs;
        x[0] = is[0];

        fft_kernel_ = fft_plans.get(x,is);
    }

private:
    void do_input_fft( real* in, complex* out, void* )
    {
        fft_iimage_->forward(in,out);
    }

    void do_output_ifft( complex* in, real* out, long_t out_elements,
                         real bias, void* stack )
    {
        real* out_scratch = reinterpret_cast<real*>(stack);

        fft_oimage_->backward(in,out_scratch);

        real scale = fft_oimage_->get_scale();

        long_t off = fft_oimage_->num_in_elements() - out_elements;

        for ( long_t i = 0; i < out_elements; ++i )
        {
            out[i] = std::max(static_cast<real>(0),
                              out_scratch[i+off] / scale + bias);
        }
    }

    void do_single_kernel( bool first,
                           real * kernel,
                           real * kernel_scratch,
                           complex * input,
                           complex * output,
                           complex * output_scratch,
                           long_t input_stride,
                           long_t output_stride )
    {
        // copy the kernel to the scratch
        long_t kernel_elements = fs_[0] * fs_[1] * fs_[2];
        std::memcpy( kernel_scratch, kernel, kernel_elements * sizeof(real));

        // append zeros
        long_t zero_bytes
            = fft_kernel_->in_memory() - kernel_elements * sizeof(real);
        std::memset( kernel_scratch + kernel_elements, 0, zero_bytes );

        // transform the kernel
        fft_kernel_->forward( kernel_scratch, output_scratch );

        // loop over the batch
        long_t n_elements = fft_kernel_->num_out_elements();

        if ( first )
        {
            for ( long_t k = 0; k < n_; ++k )
            {
                complex * a = input  + k * input_stride ;
                complex * r = output + k * output_stride;
                for ( long_t i = 0; i < n_elements; ++i )
                {
                    r[i] = a[i] * output_scratch[i];
                }
            }
        }
        else
        {
            for ( long_t k = 0; k < n_; ++k )
            {
                complex * a = input  + k * input_stride ;
                complex * r = output + k * output_stride;
                for ( long_t i = 0; i < n_elements; ++i )
                {
                    r[i] += a[i] * output_scratch[i];
                }
            }
        }

    }

    void do_single_output( long_t out_num,
                           complex* inputs,
                           complex* outputs,
                           void* stack)
    {
        long_t cstride = fft_kernel_->num_out_elements();

        complex* output_scratch = reinterpret_cast<complex*>(stack);
        real*    kernel_scratch
            = reinterpret_cast<real*>(output_scratch + cstride);

        long_t kernel_stride = fs_[0] * fs_[1] * fs_[2];

        real* first_kernel = kernel_data_ + out_num * kernel_stride * fin_;

        for ( long_t i = 0; i < fin_; ++i )
        {
            do_single_kernel( i == 0,
                              first_kernel + i * kernel_stride,
                              kernel_scratch,
                              inputs + i * cstride,
                              outputs + out_num * cstride,
                              output_scratch,
                              fin_ * cstride,
                              fout_ * cstride );
        }
    }


public:
    real * forward( real * in ) override
    {
        // do FFTs of the inputs
        complex* itransforms;

        long_t relements = is_[0] * is_[1] * is_[2];
        long_t celements = (is_[0]/2 + 1) * is_[1] * is_[2];
        itransforms = znn_malloc<complex>(n_*fin_*celements);

        for ( long_t i = 0, off = 0; i < n_; ++i )
        {
            for ( long_t j = 0; j < fin_; ++j, ++off )
            {
                handle_.add_task( &conv_layer::do_input_fft,
                                  this,
                                  in + relements * off,
                                  itransforms + celements * off );
            }
        }

        handle_.execute();

        znn_free(in);

        // collect (MADs)
        complex* otransforms = znn_malloc<complex>(n_*fout_*celements);

        for ( long_t i = 0; i < fout_; ++i )
        {
            handle_.add_task( &conv_layer::do_single_output,
                              this,
                              i,
                              itransforms,
                              otransforms );
        }

        handle_.execute( fft_kernel_->memory() );

        znn_free(itransforms);

        // do iFFT of the outputs
        vec3i  os = is_ + vec3i::one - fs_;
        long_t oelements = os[0] * os[1] * os[2];

        real* result = znn_malloc<real>(n_*fout_*oelements);

        for ( long_t i = 0, off = 0; i < n_; ++i )
        {
            for ( long_t j = 0; j < fout_; ++j, ++off )
            {
                handle_.add_task( &conv_layer::do_output_ifft,
                                  this,
                                  otransforms + celements * off,
                                  result + oelements * off,
                                  oelements,
                                  bias_data_[j] );
            }
        }

        handle_.execute( fft_oimage_->in_memory() );

        znn_free(otransforms);

        return result;
    }


};


class pooling_layer: public cpu_layer
{
private:
    task_package & handle_;

    long_t n_    ;
    long_t fin_  ;
    vec3i  is_   ;
    vec3i  fs_   ;

    int in_memory_ ;
    int out_memory_;

public:
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
    }

public:
    pooling_layer( task_package& handle,
                int n, int fin,
                vec3i const & is,
                vec3i const & fs )
        : handle_(handle)
        , n_(n)
        , fin_(fin)
        , is_(is)
        , fs_(fs)
    {
        long_t n_out = n * fs[0] * fs[1] * fs[2];

        STRONG_ASSERT( (is+vec3i::one) % fs == vec3i::zero );
        STRONG_ASSERT( fs[0] < 5 && fs[0] > 0 );
        STRONG_ASSERT( fs[1] < 5 && fs[1] > 0 );
        STRONG_ASSERT( fs[2] < 5 && fs[2] > 0 );

        vec3i os = is / fs;

        in_memory_  = n     * fin * is[0] * is[1] * is[2] * sizeof(float);
        out_memory_ = n_out * fin * os[0] * os[1] * os[2] * sizeof(float);
    }

private:
    void single_image_pool( real * im, real * out, long_t delta, void* )
    {
        vec3i strides(is_[1] * is_[2], is_[2], 1);

        vec3i is = is_;

        // Along z direction
        is[2] -= fs_[2] - 1;
        if ( fs_[2] == 2 ) pool_inplace_2( im, strides[2], is, strides );
        if ( fs_[2] == 3 ) pool_inplace_3( im, strides[2], is, strides );
        if ( fs_[2] == 4 ) pool_inplace_4( im, strides[2], is, strides );

        // Along y direction
        is[1] -= fs_[1] - 1;
        if ( fs_[1] == 2 ) pool_inplace_2( im, strides[1], is, strides );
        if ( fs_[1] == 3 ) pool_inplace_3( im, strides[1], is, strides );
        if ( fs_[1] == 4 ) pool_inplace_4( im, strides[1], is, strides );

        // Along x direction
        is[0] -= fs_[0] - 1;
        if ( fs_[0] == 2 ) pool_inplace_2( im, strides[0], is, strides );
        if ( fs_[0] == 3 ) pool_inplace_3( im, strides[0], is, strides );
        if ( fs_[0] == 4 ) pool_inplace_4( im, strides[0], is, strides );

        vec3i istrides = strides * fs_;
        vec3i size = is_ / fs_;
        vec3i ostrides( size[2] * size[1], size[2], 1 );

        for ( long_t x = 0; x < fs_[0]; ++x )
            for ( long_t y = 0; y < fs_[1]; ++y )
                for ( long_t z = 0; z < fs_[2]; ++z )
                {
                    pooling_separation( im + x*strides[0] + y*strides[1] + z*strides[2],
                                        out, istrides, ostrides, size );
                    out += delta;
                }

    }

public:
    real * forward( real * in ) override
    {
        long_t n_out = n_ * fs_[0] * fs_[1] * fs_[2];
        vec3i os = is_ / fs_;

        long_t in_elements  = is_[0] * is_[1] * is_[2];
        long_t out_elements = os[0]  * os[1]  * os[2];

        real* out = znn_malloc<real>(n_out*fin_*out_elements);

        long_t delta = out_elements * fin_ * n_;

        for ( long_t i = 0; i < fin_ * n_; ++i )
        {
            handle_.add_task( &pooling_layer::single_image_pool,
                              this,
                              in + in_elements * i,
                              out + out_elements * i,
                              delta );
        }

        handle_.execute();

        znn_free(in);
        return out;
    }


};

}}} // namespace znn::fwd::cpu3d
