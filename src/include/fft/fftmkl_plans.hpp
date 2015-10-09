//
// Copyright (C) 2012-2015  Aleksandar Zlateski <zlateski@mit.edu>
// ---------------------------------------------------------------
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.
//
#pragma once

#include <mkl_dfti.h>

#include "../types.hpp"
#include "../cube/cube.hpp"

#include <zi/utility/singleton.hpp>
#include <zi/time/time.hpp>

#include <map>
#include <unordered_map>
#include <iostream>
#include <type_traits>
#include <mutex>

namespace znn { namespace v4 {

inline vec3i fft_complex_size(const vec3i& s)
{
    auto r = s;
    r[2] /= 2;
    r[2] += 1;
    return r;
}

template< typename T >
inline vec3i fft_complex_size(const cube<T>& c)
{
    return fft_complex_size(size(c));
}

typedef DFTI_DESCRIPTOR_HANDLE* fft_plan;

class fft_plans_impl
{
private:
    std::mutex                                           m_          ;
    std::unordered_map<vec3i, fft_plan, vec_hash<vec3i>> fwd_        ;
    std::unordered_map<vec3i, fft_plan, vec_hash<vec3i>> bwd_        ;
    real                                                 time_       ;

public:
    ~fft_plans_impl()
    {
        for ( auto& p: fwd_ )
        {
            DftiFreeDescriptor(p.second);
            delete p.second;
        }

        for ( auto& p: bwd_ )
        {
            DftiFreeDescriptor(p.second);
            delete p.second;
        }
    }

    fft_plans_impl(): m_(), fwd_(), bwd_(), time_(0)
    {
    }

    fft_plan get_forward( const vec3i& s )
    {
        guard g(m_);

        fft_plan& ret = bwd_[s];

        if ( ret ) return ret;

        zi::wall_timer wt; wt.reset();

        ret = new DFTI_DESCRIPTOR_HANDLE;

        MKL_LONG status, l[3];
        MKL_LONG strides_out[4];
        MKL_LONG strides_in[4];

        l[0] = s[0]; l[1] = s[1]; l[2] = s[2];

        auto fs = fft_complex_size(s);

        strides_out[0] = 0; strides_out[1] = fs[1]*fs[2];
        strides_out[2] = fs[2]; strides_out[3] = 1;

        strides_in[0] = 0; strides_in[1] = s[1]*s[2];
        strides_in[2] = s[2]; strides_in[3] = 1;

#ifdef ZNN_USE_FLOATS
        status = DftiCreateDescriptor( ret, DFTI_SINGLE, DFTI_REAL, 3, l );
#else
        status = DftiCreateDescriptor( ret, DFTI_DOUBLE, DFTI_REAL, 3, l );
#endif
        status = DftiSetValue( *ret ,
                               DFTI_CONJUGATE_EVEN_STORAGE,
                               DFTI_COMPLEX_COMPLEX );

        status = DftiSetValue( *ret, DFTI_PLACEMENT, DFTI_NOT_INPLACE );
        status = DftiSetValue( *ret, DFTI_OUTPUT_STRIDES, strides_out );
        status = DftiSetValue( *ret, DFTI_INPUT_STRIDES, strides_in );

        status = DftiCommitDescriptor(*ret);

        time_ += wt.elapsed<real>();

//        std::cout << "Total time spent creating fft plans: "
//                  << time_ << std::endl;

        return ret;
    }

    fft_plan get_backward( const vec3i& s )
    {
        guard g(m_);

        fft_plan& ret = fwd_[s];

        if ( ret ) return ret;

        zi::wall_timer wt; wt.reset();

        ret = new DFTI_DESCRIPTOR_HANDLE;

        MKL_LONG status, l[3];
        MKL_LONG strides_out[4];
        MKL_LONG strides_in[4];

        l[0] = s[0]; l[1] = s[1]; l[2] = s[2];

        auto fs = fft_complex_size(s);

        strides_out[0] = 0; strides_out[1] = fs[1]*fs[2];
        strides_out[2] = fs[2]; strides_out[3] = 1;

        strides_in[0] = 0; strides_in[1] = s[1]*s[2];
        strides_in[2] = s[2]; strides_in[3] = 1;

#ifdef ZNN_USE_FLOATS
        status = DftiCreateDescriptor( ret, DFTI_SINGLE, DFTI_REAL, 3, l );
#else
        status = DftiCreateDescriptor( ret, DFTI_DOUBLE, DFTI_REAL, 3, l );
#endif
        status = DftiSetValue( *ret ,
                               DFTI_CONJUGATE_EVEN_STORAGE,
                               DFTI_COMPLEX_COMPLEX );

        status = DftiSetValue( *ret, DFTI_PLACEMENT, DFTI_NOT_INPLACE );
        status = DftiSetValue( *ret, DFTI_OUTPUT_STRIDES, strides_in );
        status = DftiSetValue( *ret, DFTI_INPUT_STRIDES, strides_out );

        status = DftiCommitDescriptor(*ret);

        time_ += wt.elapsed<real>();

//        std::cout << "Total time spent creating fft plans: "
//                  << time_ << std::endl;

        return ret;
    }

}; // class fft_plans_impl

namespace {
fft_plans_impl& fft_plans =
    zi::singleton<fft_plans_impl>::instance();
} // anonymous namespace


}} // namespace znn::v4
