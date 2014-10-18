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

#ifndef ZNN_VOLUME_POOL_HPP_INCLUDED
#define ZNN_VOLUME_POOL_HPP_INCLUDED

#include "types.hpp"

#include <boost/shared_ptr.hpp>

#include <zi/utility/assert.hpp>
#include <zi/utility/for_each.hpp>
#include <zi/utility/singleton.hpp>
#include <zi/concurrency.hpp>

#include <map>
#include <list>
#include <iostream>

namespace zi {
namespace znn {

class observable_pool
{
public:
    virtual ~observable_pool() {}
    virtual std::size_t mem_used() const = 0;
    virtual std::size_t mem_cache() const = 0;
    virtual std::size_t cache_size() const = 0;
    virtual std::size_t total_size() const = 0;
    virtual std::size_t reduce_cache(std::size_t) = 0;
}; // class observable_pool


class cache_manager_impl
{
private:

    typedef std::map<std::size_t
                     , observable_pool*
                     , std::greater<size_t> >    pools_map;

private:
    zi::mutex   m_;
    std::size_t max_memory_;
    std::size_t used_memory_;
    std::size_t cached_memory_;
    pools_map   map_;

public:
    cache_manager_impl()
        : m_()
        , max_memory_(static_cast<std::size_t>(48)*1024*1024*1024)
        , used_memory_(0)
        , cached_memory_(0)
        , map_()
    {}

    void add_used(std::size_t s)
    {
        zi::mutex::guard g(m_);
        used_memory_ += s;
    }

    void add_cached(std::size_t s)
    {
        zi::mutex::guard g(m_);
        cached_memory_ += s;
    }

    void reduce_cached(std::size_t s)
    {
        zi::mutex::guard g(m_);
        cached_memory_ -= s;
    }

    void report()
    {
        zi::mutex::guard g(m_);
        std::cout << "Used: " << (used_memory_>>20) << " MB\n"
                  << "Cash: " << (cached_memory_>>20) << " MB" << std::endl;
    }


}; //  class cache_manager

namespace {
cache_manager_impl& cache_manager =
    zi::singleton<cache_manager_impl>::instance();
} // anonymous namespace

namespace detail {

template<class V>
class volume_pool: public observable_pool
{
private:
    vec3i          size_;
    std::list<V*>  list_;
    zi::mutex      m_   ;

    std::size_t    mem_used_;
    std::size_t    cached_memory_;
    std::size_t    num_elements_;

public:
    void clear()
    {
        zi::mutex::guard g(m_);
        FOR_EACH( it, list_ )
        {
            //std::cout << "Destroyed Volume\n";
            delete (*it);
        }
        list_.clear();
        mem_used_ = 0;
    }

public:
    void return_volume( V* v )
    {
        zi::mutex::guard g(m_);
        //std::cout << "Returned Volume\n";
        list_.push_back(v);
        typedef typename V::element element_type;
        cached_memory_ +=
            v->shape()[0]*
            v->shape()[1]*
            v->shape()[2]*sizeof(element_type);
        cache_manager.add_cached(
            v->shape()[0]*
            v->shape()[1]*
            v->shape()[2]*sizeof(element_type));
    }

public:
    volume_pool( const vec3i& s )
        : size_(s)
        , list_()
        , m_()
        , mem_used_(0)
        , cached_memory_(0)
        , num_elements_(0)
    { }

    ~volume_pool()
    {
        clear();
    }

    boost::shared_ptr<V> get()
    {
        V* r = 0;
        {
            zi::mutex::guard g(m_);
            if ( list_.size() > 0 )
            {
                r = list_.back();
                list_.pop_back();
                typedef typename V::element element_type;
                cached_memory_ -= size_[0]*size_[1]*size_[2]*sizeof(element_type);
                cache_manager.reduce_cached(
                    size_[0]*size_[1]*size_[2]*sizeof(element_type));
            }
        }

        if ( !r )
        {
            //std::cout << "Created Volume\n";
            r = new V(boost::extents[size_[0]][size_[1]][size_[2]]);
            {
                zi::mutex::guard g(m_);
                typedef typename V::element element_type;
                cache_manager.add_used(
                    size_[0]*size_[1]*size_[2]*sizeof(element_type));
                mem_used_ += size_[0]*size_[1]*size_[2]*sizeof(element_type);
                ++num_elements_;
            }
        }

        return boost::shared_ptr<V>(r, zi::bind(&volume_pool::return_volume,
                                                this, zi::placeholders::_1));
    }

    std::size_t reduce_cache(std::size_t n = 1)
    {
        zi::mutex::guard g(m_);

        for ( std::size_t i = 0; i < n; ++i )
        {
            if ( list_.size() > 0 )
            {
                V* r = list_.back();
                list_.pop_back();
                typedef typename V::element element_type;
                cached_memory_ -= size_[0]*size_[1]*size_[2]*sizeof(element_type);
                --num_elements_;
                delete r;
            }
        }

        return cached_memory_;
    }

    std::size_t mem_used() const
    {
        zi::mutex::guard g(m_);
        return mem_used_;
    }

    std::size_t mem_cache() const
    {
        zi::mutex::guard g(m_);
        return cached_memory_;
    }

    std::size_t cache_size() const
    {
        zi::mutex::guard g(m_);
        return list_.size();
    }

    std::size_t total_size() const
    {
        zi::mutex::guard g(m_);
        return num_elements_;
    }

}; // class volume_pool

class volume_pools
{
private:
    zi::mutex                                  double3d_m_     ;
    std::map< vec3i, volume_pool<double3d>* >  double3d_pools_ ;

    zi::mutex                                  complex3d_m_    ;
    std::map< vec3i, volume_pool<complex3d>* > complex3d_pools_;

    // [07/08/2013 kisuklee]
    zi::mutex                                  bool3d_m_    ;
    std::map< vec3i, volume_pool<bool3d>* >    bool3d_pools_;

    zi::mutex                                  long3d_m_    ;
    std::map< vec3i, volume_pool<long3d>* >    long3d_pools_;


public:
    std::size_t mem_used() const
    {
        std::size_t r = 0;
        {
            zi::mutex::guard g(double3d_m_);
            FOR_EACH( it, double3d_pools_ )
            {
                r += it->second->mem_used();
            }
        }
        {
            zi::mutex::guard g(complex3d_m_);
            FOR_EACH( it, complex3d_pools_ )
            {
                r += it->second->mem_used();
            }
        }
        {
            zi::mutex::guard g(bool3d_m_);
            FOR_EACH( it, bool3d_pools_ )
            {
                r += it->second->mem_used();
            }
        }
        {
            zi::mutex::guard g(long3d_m_);
            FOR_EACH( it, long3d_pools_ )
            {
                r += it->second->mem_used();
            }
        }
        return r;
    }

    std::size_t cache_size() const
    {
        std::size_t r = 0;
        {
            zi::mutex::guard g(double3d_m_);
            FOR_EACH( it, double3d_pools_ )
            {
                r += it->second->cache_size();
            }
        }
        {
            zi::mutex::guard g(complex3d_m_);
            FOR_EACH( it, complex3d_pools_ )
            {
                r += it->second->cache_size();
            }
        }
        {
            zi::mutex::guard g(bool3d_m_);
            FOR_EACH( it, bool3d_pools_ )
            {
                r += it->second->cache_size();
            }
        }
        {
            zi::mutex::guard g(long3d_m_);
            FOR_EACH( it, long3d_pools_ )
            {
                r += it->second->cache_size();
            }
        }
        return r;
    }

private:
    void clear_double3d()
    {
        zi::mutex::guard g(double3d_m_);
        FOR_EACH( it, double3d_pools_ )
        {
            it->second->clear();
        }
    }

    void clear_complex3d()
    {
        zi::mutex::guard g(complex3d_m_);
        FOR_EACH( it, complex3d_pools_ )
        {
            it->second->clear();
        }
    }

    // [07/08/2013 kisuklee]
    void clear_bool3d()
    {
        zi::mutex::guard g(bool3d_m_);
        FOR_EACH( it, bool3d_pools_ )
        {
            it->second->clear();
        }
    }

    void clear_long3d()
    {
        zi::mutex::guard g(long3d_m_);
        FOR_EACH( it, long3d_pools_ )
        {
            it->second->clear();
        }
    }

    void destroy_double3d()
    {
        zi::mutex::guard g(double3d_m_);
        FOR_EACH( it, double3d_pools_ )
        {
            delete it->second;
        }
    }

    void destroy_complex3d()
    {
        zi::mutex::guard g(complex3d_m_);
        FOR_EACH( it, complex3d_pools_ )
        {
            delete it->second;
        }
    }

    // [07/08/2013 kisuklee]
    void destroy_bool3d()
    {
        zi::mutex::guard g(bool3d_m_);
        FOR_EACH( it, bool3d_pools_ )
        {
            delete it->second;
        }
    }

    void destroy_long3d()
    {
        zi::mutex::guard g(long3d_m_);
        FOR_EACH( it, long3d_pools_ )
        {
            delete it->second;
        }
    }

    volume_pool<double3d>* get_double3d_pool( const vec3i& s )
    {
        zi::mutex::guard g(double3d_m_);
        if ( double3d_pools_.count(s) )
        {
            return double3d_pools_[s];
        }
        else
        {
            volume_pool<double3d>* r = new volume_pool<double3d>(s);
            double3d_pools_[s] = r;
            return r;
        }
    }

    volume_pool<complex3d>* get_complex3d_pool( const vec3i& s )
    {
        zi::mutex::guard g(complex3d_m_);
        if ( complex3d_pools_.count(s) )
        {
            return complex3d_pools_[s];
        }
        else
        {
            volume_pool<complex3d>* r = new volume_pool<complex3d>(s);
            complex3d_pools_[s] = r;
            return r;
        }
    }

    // [07/08/2013 kisuklee]
    volume_pool<bool3d>* get_bool3d_pool( const vec3i& s )
    {
        zi::mutex::guard g(bool3d_m_);
        if ( bool3d_pools_.count(s) )
        {
            return bool3d_pools_[s];
        }
        else
        {
            volume_pool<bool3d>* r = new volume_pool<bool3d>(s);
            bool3d_pools_[s] = r;
            return r;
        }
    }

    volume_pool<long3d>* get_long3d_pool( const vec3i& s )
    {
        zi::mutex::guard g(long3d_m_);
        if ( long3d_pools_.count(s) )
        {
            return long3d_pools_[s];
        }
        else
        {
            volume_pool<long3d>* r = new volume_pool<long3d>(s);
            long3d_pools_[s] = r;
            return r;
        }
    }

public:
    volume_pools()
    {}

    ~volume_pools()
    {
        destroy_double3d();
        destroy_complex3d();
        destroy_bool3d(); // [07/08/2013 kisuklee]
        destroy_long3d(); // [07/08/2013 kisuklee]
    }

    shared_ptr<double3d>
    get_double3d( std::size_t x, std::size_t y, std::size_t z )
    {
        return get_double3d_pool(vec3i(x,y,z))->get();
    }

    shared_ptr<double3d>
    get_double3d( shared_ptr<double3d> x )
    {
        return get_double3d(x->shape()[0], x->shape()[1], x->shape()[2] );
    }

    shared_ptr<double3d>
    get_double3d( const vec3i& v )
    {
        return get_double3d_pool(v)->get();
    }

    shared_ptr<complex3d>
    get_complex3d( std::size_t x, std::size_t y, std::size_t z )
    {
        return get_complex3d_pool(vec3i(x,y,z/2+1))->get();
    }

    shared_ptr<complex3d>
    get_complex3d( shared_ptr<complex3d> x )
    {
        return get_complex3d_pool(vec3i
                                  (x->shape()[0],x->shape()[1],x->shape()[2]))->get();
    }

    shared_ptr<complex3d>
    get_complex3d( const vec3i& v )
    {
        vec3i vc(v[0], v[1], (v[2]/2)+1);
        return get_complex3d_pool(vc)->get();
    }

    shared_ptr<complex3d>
    get_complex3d( shared_ptr<double3d> x )
    {
        return get_complex3d(x->shape()[0], x->shape()[1], x->shape()[2] );
    }

    // [07/08/2013 kisuklee]
    shared_ptr<bool3d>
    get_bool3d( std::size_t x, std::size_t y, std::size_t z )
    {
        return get_bool3d_pool(vec3i(x,y,z))->get();
    }

    // [07/08/2013 kisuklee]
    shared_ptr<bool3d>
    get_bool3d( shared_ptr<bool3d> x )
    {
        return get_bool3d(x->shape()[0], x->shape()[1], x->shape()[2] );
    }

    // [07/08/2013 kisuklee]
    shared_ptr<bool3d>
    get_bool3d( const vec3i& v )
    {
        return get_bool3d_pool(v)->get();
    }

    shared_ptr<long3d>
    get_long3d( std::size_t x, std::size_t y, std::size_t z )
    {
        return get_long3d_pool(vec3i(x,y,z))->get();
    }

    shared_ptr<long3d>
    get_long3d( shared_ptr<long3d> x )
    {
        return get_long3d(x->shape()[0], x->shape()[1], x->shape()[2] );
    }

    shared_ptr<long3d>
    get_long3d( shared_ptr<double3d> x )
    {
        return get_long3d(x->shape()[0], x->shape()[1], x->shape()[2] );
    }

    shared_ptr<long3d>
    get_long3d( const vec3i& v )
    {
        return get_long3d_pool(v)->get();
    }

    void clear()
    {
        clear_double3d();
        clear_complex3d();
        clear_bool3d(); // [07/08/2013 kisuklee]
        clear_long3d(); // [07/08/2013 kisuklee]
    }

}; // class volume_pools

} // namespace detail

namespace {
detail::volume_pools& volume_pools =
    zi::singleton<detail::volume_pools>::instance();
detail::volume_pools& volume_pool =
    zi::singleton<detail::volume_pools>::instance();
} // anonymous namespace

}} // namespace zi::znn

#endif // ZNN_VOLUME_POOL_HPP_INCLUDED
