#pragma once

#include "task_package.hpp"
#include "../types.hpp"
#include "../assert.hpp"
#include "fft.hpp"
#include "conv.hpp"
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

    int in_memory_ ;
    int out_memory_;

    real * kernel_data_ ;
    real * bias_data_   ;

    fft_transformer* fft_;

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
    {
        fft_ = fft_plans.get(is,fs);

        kernel_data_ = znn_malloc<real>(fin * fout * fft_->kernel_elements());
        bias_data_   = znn_malloc<real>(fout);

        in_memory_ = n * fin * fft_->image_memory();
        out_memory_ = n * fout * fft_->result_memory();
    }

private:
    void do_input_fft( real* in, complex* out, void* )
    {
        fft_->forward_image(in,out);
    }

    void do_input_fft_padded( real* in, complex* out, void* scratch)
    {
        // copy the image
        real* tmp = reinterpret_cast<real*>(scratch);
        std::memcpy( tmp, in, fft_->image_memory());

        // append zeros
        long_t zero_bytes = fft_->image_scratch_memory()
            - fft_->image_memory();
        std::memset( tmp + fft_->image_elements(), 0, zero_bytes );

        fft_->forward_image(tmp,out);
    }

    void do_output_ifft( complex* in, real* out, long_t out_elements,
                         real bias, void* stack )
    {
        real* out_scratch = reinterpret_cast<real*>(stack);

        fft_->backward(in,out_scratch);
        real scale = fft_->get_scale();

        long_t off = fft_->result_scratch_elements() - out_elements;

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
        std::memcpy( kernel_scratch, kernel, fft_->kernel_memory());

        // append zeros
        long_t zero_bytes = fft_->kernel_scratch_memory()
            - fft_->kernel_memory();
        std::memset( kernel_scratch + fft_->kernel_elements(),
                     0, zero_bytes );

        // transform the kernel
        fft_->forward_kernel( kernel_scratch, output_scratch );

        // loop over the batch
        long_t n_elements = fft_->transform_elements();

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
                           void*)
    {
        long_t cstride = fft_->transform_elements();

        complex* output_scratch
            = znn_malloc<complex>(fft_->transform_elements());
        real*    kernel_scratch
            = znn_malloc<real>   (fft_->kernel_scratch_elements());

        long_t kernel_stride = fft_->kernel_elements();

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


private:
    complex * transform_inputs( real * in )
    {
        long_t   relements   = fft_->image_elements();
        long_t   celements   = fft_->transform_elements();
        complex* itransforms = znn_malloc<complex>(n_*fin_*celements);

        if ( fft_->needs_padding() )
        {
            for ( long_t i = 0, off = 0; i < n_; ++i )
                for ( long_t j = 0; j < fin_; ++j, ++off )
                {
                    handle_.add_task( &conv_layer::do_input_fft_padded,
                                      this,
                                      in + relements * off,
                                      itransforms + celements * off );
                }

            handle_.execute(fft_->image_scratch_memory());
        }
        else
        {
            for ( long_t i = 0, off = 0; i < n_; ++i )
                for ( long_t j = 0; j < fin_; ++j, ++off )
                {
                    handle_.add_task( &conv_layer::do_input_fft,
                                      this,
                                      in + relements * off,
                                      itransforms + celements * off );
                }

            handle_.execute();
        }

        znn_free(in);
        return itransforms;
    }

    complex * collect_outputs( complex * itransforms )
    {
        long_t   celements   = fft_->transform_elements();
        complex* otransforms = znn_malloc<complex>(n_*fout_*celements);

        for ( long_t i = 0; i < fout_; ++i )
        {
            handle_.add_task( &conv_layer::do_single_output,
                              this,
                              i,
                              itransforms,
                              otransforms );
        }

        handle_.execute();
        znn_free(itransforms);

        return otransforms;
    }

    real * process_outputs( complex * otransforms )
    {
        long_t celements = fft_->transform_elements();
        long_t oelements = fft_->result_elements();

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

        handle_.execute( fft_->result_scratch_memory() );

        znn_free(otransforms);

        return result;
    }

public:
    real * forward( real * in ) override
    {
        // do FFTs of the inputs
        complex* itransforms = transform_inputs(in);
        complex* otransforms = collect_outputs(itransforms);
        return process_outputs(otransforms);
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


class direct_conv_layer: public cpu_layer
{
private:
    task_package & handle_;

    long_t n_    ;
    long_t fin_  ;
    long_t fout_ ;

    vec3i is_;
    vec3i ks_;
    vec3i os_;

    int in_memory_ ;
    int out_memory_;

    real * kernel_data_ ;
    real * bias_data_   ;

    convolver* conv_;

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

    ~direct_conv_layer()
    {
        znn_free(kernel_data_);
        znn_free(bias_data_);
        delete conv_;
    }

public:
    direct_conv_layer( task_package& handle,
                       int n, int fin, int fout,
                       vec3i const & is,
                       vec3i const & fs )
        : handle_(handle)
        , n_(n)
        , fin_(fin)
        , fout_(fout)
        , is_(is)
        , ks_(fs)
        , os_(is - fs + vec3i::one)
    {
        conv_ = new convolver(is,fs);

        kernel_data_ = znn_malloc<real>(fin * fout * fs[0]*fs[1]*fs[2]);
        bias_data_   = znn_malloc<real>(fout);

        vec3i os = is + vec3i::one - fs;

        in_memory_ = n * fin * is[0] * is[1] * is[2] * sizeof(real);
        out_memory_ = n * fout * os[0] * os[1] * os[2] * sizeof(real);
    }

private:

#if defined(ZNN_USE_MKL_CONVOLUTION)
    void do_single_output( real* input , long_t istride,
                           real* kernel, long_t kstride,
                           real* out, long_t oelements,
                           real bias, void* stack)
    {
        conv_->convolve_add(input, kernel, out);

        real* tmp = reinterpret_cast<real*>(stack);

        for ( long_t i = 0; i < fin_; ++i )
        {
            conv_->convolve_add(input + i * istride, kernel + i * kstride, tmp);
            for ( long_t j = 0; j < oelements; ++j ) out[j] += tmp[j];
        }

        for ( long_t i = 0; i < oelements; ++i )
        {
            out[i] = std::max(static_cast<real>(0), out[i] + bias);
        }
    }
#else
    void do_single_output( real* input , long_t istride,
                           real* kernel, long_t kstride,
                           real* out, long_t oelements,
                           real bias, void* )
    {
        conv_->convolve(input, kernel, out);
        for ( long_t i = 1; i < fin_; ++i )
        {
            conv_->convolve_add(input + i * istride, kernel + i * kstride, out);
        }

        for ( long_t i = 0; i < oelements; ++i )
        {
            out[i] = std::max(static_cast<real>(0), out[i] + bias);
        }
    }
#endif

public:
    real * forward( real * in ) override
    {
        long_t istride = is_[0] * is_[1] * is_[2];
        long_t ostride = os_[0] * os_[1] * os_[2];
        long_t kstride = ks_[0] * ks_[1] * ks_[2];

        long_t nistride = fin_  * istride;
        long_t nostride = fout_ * ostride;

        real* out = znn_malloc<real>(n_ * nostride);

        for ( long_t n = 0; n < n_; ++n )
        {
            for ( long_t i = 0; i < fout_; ++i )
            {
                real* first_kernel = kernel_data_ + i * kstride * fin_;
                handle_.add_task( &direct_conv_layer::do_single_output,
                                  this,
                                  in + n * nistride, istride,
                                  first_kernel, kstride,
                                  out + n * nostride + i * ostride, ostride,
                                  bias_data_[i] );

            }
        }

#if defined(ZNN_USE_MKL_CONVOLUTION)
        handle_.execute(os_[0]*os_[1]*os_[2]*sizeof(real));
#else
        handle_.execute();
#endif

        znn_free(in);
        return out;
    }


};



}}} // namespace znn::fwd::cpu3d
