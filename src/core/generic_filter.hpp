//
// Copyright (C) 2014  Aleksandar Zlateski <zlateski@mit.edu>
// ----------------------------------------------------------
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

#ifndef ZNN_GENERIC_FILTER_HPP_INCLUDED
#define ZNN_GENERIC_FILTER_HPP_INCLUDED

#include "types.hpp"
#include "volume_pool.hpp"

#include <utility>
#include <algorithm>
#include <set>
#include <zi/utility/singleton.hpp>

namespace zi {
namespace znn {

namespace detail {

template<class C>
struct pair_comparator_impl
{
private:
    C f;

public:
    pair_comparator_impl()
        : f()
    {}

    explicit pair_comparator_impl(const C& c)
        : f(c)
    {}

    bool operator()(const std::pair<double,int64_t>& a,
                    const std::pair<double,int64_t>& b) const
    {
        if ( f(a.first,b.first) )
        {
            return true;
        }
        else if ( f(b.first,a.first) )
        {
            return false;
        }
        else
        {
            return a.second < b.second;
        }
    }
}; // class pair_comparator_impl

} // namespace detail


template< class F >
inline void linear_generic_filter_pass(
    double*, double*, int64_t*, std::ptrdiff_t, std::size_t, const F& )
    __attribute__((always_inline));


template< class F >
inline void
linear_generic_filter_pass( double* first,
                            double* last,
                            int64_t* ip,
                            std::ptrdiff_t delta,
                            std::size_t size,
                            const F& compare_function )
{
    typedef std::pair<double, int64_t>         value_type  ;
    typedef F                                  compare_type;
    typedef std::set<value_type,compare_type>  set_type    ;

    double* dhead = first;
    double* dtail = first;
    int64_t* ihead = ip;
    int64_t* itail = ip;

    set_type set(compare_function);

    ZI_ASSERT((first+(size-1)*delta)<=last);

    for ( std::size_t i = 0; i<size-1; ++i, dtail += delta, itail += delta )
    {
        set.insert(value_type(*dtail,*itail));
    }

    ZI_ASSERT((dtail-delta)<=last);

    while (dtail <= last)
    {
        set.insert(value_type(*dtail, *itail));
        value_type r = *set.begin();

        set.erase(value_type(*dhead, *ihead));

        *dhead = r.first ;
        *ihead = r.second;

        dtail += delta;
        itail += delta;
        dhead += delta;
        ihead += delta;
    }
}

template< class F >
inline std::pair<double3d_ptr, long3d_ptr>
generic_filter(double3d_ptr v,
               std::size_t fx,
               std::size_t fy,
               std::size_t fz,
               std::size_t sx,
               std::size_t sy,
               std::size_t sz,
               const F& double_compare_function = F())
{
    std::size_t xs = v->shape()[0];
    std::size_t ys = v->shape()[1];
    std::size_t zs = v->shape()[2];

    std::size_t n = xs*ys*zs;

    typedef std::pair<double, int64_t>         value_type  ;
    typedef detail::pair_comparator_impl<F>    compare_type;

    double3d_ptr vtmp = volume_pool.get_double3d(v);
    long3d_ptr   itmp = volume_pool.get_long3d(v);

    compare_type compare_function(double_compare_function);

    *vtmp = *v;

    for ( std::size_t i = 0; i < n; ++i )
    {
        itmp->data()[i] = static_cast<int64_t>(i);
    }

    std::size_t dx = ys*zs;    

    // compute along z-axis
    if ( fz > 1 )
    {
        for ( std::size_t x = 0; x < xs; ++x )
        {
            for ( std::size_t y = 0; y < ys; ++y )
            {
                for ( std::size_t z = 0; z < sz; ++z )
                {
                    linear_generic_filter_pass(
                        &((*vtmp)[x][y][z]),
                        &((*vtmp)[x][y][zs-1]),
                        &((*itmp)[x][y][z]),
                        static_cast<std::ptrdiff_t>(sz),
                        fz,
                        compare_function);
                }
            }
        }
    }

    // compute along y-axis
    if ( fy > 1 )
    {
        for ( std::size_t x = 0; x < xs; ++x )
        {
            for ( std::size_t z = 0; z < zs; ++z )
            {
                for ( std::size_t y = 0; y < sy; ++y )
                {
                    linear_generic_filter_pass(
                        &((*vtmp)[x][y][z]),
                        &((*vtmp)[x][ys-1][z]),
                        &((*itmp)[x][y][z]),
                        // static_cast<std::ptrdiff_t>(sy*xs),
                        static_cast<std::ptrdiff_t>(sy*zs),
                        fy,
                        compare_function);
                }
            }
        }
    }

    // compute along x-axis
    if ( fx > 1 )
    {
        for ( std::size_t y = 0; y < ys; ++y )
        {
            for ( std::size_t z = 0; z < zs; ++z )
            {
                for ( std::size_t x = 0; x < sx; ++x )
                {
                    linear_generic_filter_pass(
                        &((*vtmp)[x][y][z]),
                        &((*vtmp)[xs-1][y][z]),
                        &((*itmp)[x][y][z]),
                        static_cast<std::ptrdiff_t>(sx*dx),
                        fx,
                        compare_function);
                }
            }
        }
    }

    std::size_t rfx = (fx-1)*sx+1;
    std::size_t rfy = (fy-1)*sy+1;
    std::size_t rfz = (fz-1)*sz+1;

    return std::pair<double3d_ptr, long3d_ptr>
        ( volume_utils::crop(vtmp,xs+1-rfx,ys+1-rfy,zs+1-rfz),
          volume_utils::crop(itmp,xs+1-rfx,ys+1-rfy,zs+1-rfz) );
}

template< class F >
inline std::pair<double3d_ptr, long3d_ptr>
generic_filter(double3d_ptr v,
               const vec3i& f,
               const vec3i& s,
               const F& fn = F())
{
    return generic_filter<F>(v,
                             f[0], f[1], f[2],
                             s[0], s[1], s[2],
                             fn);
}

inline double3d_ptr do_filter_backprop(double3d_ptr v,
                                       long3d_ptr p,
                                       std::size_t fx,
                                       std::size_t fy,
                                       std::size_t fz)
{
    std::size_t xs = v->shape()[0];
    std::size_t ys = v->shape()[1];
    std::size_t zs = v->shape()[2];

    std::size_t n = xs*ys*zs;
    double3d_ptr r = volume_pool.get_double3d(xs + fx - 1,
                                              ys + fy - 1,
                                              zs + fz - 1);
    volume_utils::zero_out(r);

    int64_t* pp = p->data();
    double*  vp = v->data();
    double*  rp = r->data();

    for ( std::size_t i = 0; i < n; ++i )
    {
        rp[pp[i]] += vp[i];
    }

    return r;
}

inline double3d_ptr do_filter_backprop(double3d_ptr v,
                                       long3d_ptr p,
                                       const vec3i& f)
{
    return do_filter_backprop(v,p,f[0],f[1],f[2]);
}


}} // namespace zi::znn

#endif // ZNN_GENERIC_FILTER_HPP_INCLUDED
