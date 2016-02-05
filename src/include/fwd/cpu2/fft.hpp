#pragma once

#if defined(ZNN_USE_MKL_FFT)
#  include "fft/fftmkl.hpp"
#else
#  include "fft/fftw.hpp"
#endif

#include <map>
#include <mutex>
#include <zi/utility/singleton.hpp>

namespace znn { namespace fwd {

class fft_plans_impl
{
private:
    std::mutex                                           m_  ;
    std::map<vec3i, std::map<vec3i, fft_transformer*>>   map_;

public:
    ~fft_plans_impl()
    {
        for ( auto & p: map_ )
            for ( auto & q: p.second )
                delete q.second;
    }

    fft_plans_impl(): m_(), map_()
    {
    }

    fft_transformer* get( vec3i const & r, vec3i const & c )
    {
        typedef  fft_transformer* ret_type;

        guard g(m_);

        ret_type& ret = map_[r][c];
        if ( ret ) return ret;

        ret = new fft_transformer(r,c);
        return ret;
    }

}; // class fft_plans_impl


namespace {
fft_plans_impl& fft_plans =
    zi::singleton<fft_plans_impl>::instance();
} // anonymous namespace

}} // namespace znn::fwd
