//
// Copyright (C) 2015-present  Aleksandar Zlateski <zlateski@mit.edu>
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

#ifndef ZNN_CORE_EDGE_HPP_INCLUDED
#define ZNN_CORE_EDGE_HPP_INCLUDED

#include "types.hpp"
#include "fftw.hpp"
#include "utils.hpp"
#include "filter.hpp"

namespace zi {
namespace znn {

/****************************************************************************

       +------------------+
       |     created      |
       +--------+---------+
                |
       +--------v---------+
       |  uninitialized   |
       +--------+---------+
                |
       +--------v---------+        +------------------+
       |   initializing   |        |     updating     <-------+
       +---+----+---------+        +---------+----+---+       |
           |    |                            |    |           |
           |    |    +------------------+    |    |           |
           |    +---->      ready       <----+    |           |
           |         +-+------+-------+-+         |           |
           |           |      |       |           |           |
     +-----v-----------v+     |      +v-----------v-----+     |
     |  init_and_fwd    |     |      |  update_and_fwd  |     |
     +--------+---------+     |      +----------+-------+     |
              |               |                 |             |
              |      +--------v---------+       |             |
              +------>     fwd_done     <-------+             |
                     +--------+---------+                     |
                              |                               |
                     +--------v---------+                     |
                     |     bwd_done     +---------------------+
                     +------------------+

****************************************************************************/

class znode;

class transfer_function_interface
{
public:
    virtual void apply(vol<double>&) noexcept = 0;
    virtual void apply(vol<double>&, double) noexcept = 0;
    virtual void apply_grad(vol<double>&, vol<double>&) noexcept = 0;
};

template<class F>
class transfer_function_wrapper: public transfer_function_interface
{
private:
    F f_;

public:
    explicit transfer_function_wrapper(F f = F())
        : f_(f)
    {}

    void apply(vol<double>& v) noexcept override
    {
        double* d = v.data();
        size_t  n = v.num_elements();
        for ( size_t i = 0; i < n; ++i )
            d[i] = f_(d[i]);
    }

    void apply(vol<double>& v, double bias) noexcept override
    {
        double* d = v.data();
        size_t  n = v.num_elements();
        for ( size_t i = 0; i < n; ++i )
            d[i] = f_(d[i] + bias);
    }

    void apply_grad(vol<double>& g, vol<double>& f) noexcept override
    {
        ZI_ASSERT(size(g)==size(f));
        double* gp = g.data();
        double* fp = f.data();
        size_t  n = g.num_elements();
        for ( size_t i = 0; i < n; ++i )
            gp[i] *= f_.grad(gp[i]);
    }
};

class transfer_function
{
private:
    std::unique_ptr<transfer_function_interface> f_;

public:
    transfer_function()
        : f_()
    {}

    template<typename F>
    explicit transfer_function(F f)
        : f_(new transfer_function_wrapper<F>(f))
    {}

    template<typename F>
    transfer_function& operator=(F f)
    {
        f_ = std::unique_ptr<transfer_function_interface>
            (new transfer_function_wrapper<F>(f));
        return *this;
    }

    void apply(vol<double>& v) noexcept
    {
        f_->apply(v);
    }

    void apply(vol<double>& v, double bias) noexcept
    {
        f_->apply(v, bias);
    }

    void apply_grad(vol<double>& g, vol<double>& f) noexcept
    {
        f_->apply_grad(g,f);
    }
};

struct sigmoid
{
    inline double operator()(double x) const noexcept
    {
        return static_cast<double>(1) / (static_cast<double>(1) + std::exp(-x));
    }

    inline double grad(double f) const noexcept
    {
        return f * (static_cast<double>(1) - f);
    }

}; // struct sigmoid

template< class Net, class In, class Out >
class edge_base
{
protected:
    Net*    net_;
    In*     in_ ;
    Out*    out_;
    filter* W_;

    std::mutex mutex_;

    vec3i size_    ;
    vec3i sparse_  ;
    vec3i in_size_ ;
    vec3i out_size_;

    edge_base( Net* net, In* in, Out* out, filter* W,
               const vec3i& size, const vec3i& sparse, const vec3i& in_size)
        : net_(net)
        , in_(in)
        , out_(out)
        , W_(W)
        , size_(size)
        , sparse_(sparse)
        , in_size_(in_size)
        , out_size_(in_size-(size-vec3i::one)*sparse)
    {}
};

template< class Net, class In, class Out >
class simple_edge: edge_base<Net,In,Out>
{
private:
    typedef edge_base<Net,In,Out> base;

};

class znode;

class zedge
{
private:
    enum class state
    {
        created,
        uninitialized,
        initializing,
        init_and_fwd,
        fwd_done,
        bwd_done,
        updating,
        update_and_fwd,
        ready
    };

    struct weight_update_task
    {
        vol_p<double>  F;
        vol_p<complex> F_fft;
        vol_p<double>  G;
        vol_p<complex> G_fft;
    };

    struct initialization_task {};

private:
    std::mutex mutex_;

    znode* in_ ;
    znode* out_;

    state  state_  = state::created;
    bool   is_fft_ = false;

    vec3i  size_  ;
    vec3i  sparse_;

    vec3i  in_size_ ;
    vec3i  out_size_;

    vol_p<double>  W_;
    vol_p<complex> W_fft_;

    vol_p<double>  F_in_;
    vol_p<complex> F_in_fft_;

    // owned by the thread possibly doing the update/init
    weight_update_task*  update_task_     = nullptr;
    initialization_task* initialize_task_ = nullptr;

    // weight update stuff
    double        eta_;

    // weight update additional
    double        momenum_      ;
    vol_p<double> mom_volume_   ;
    double        weight_decay_ ;
    std::size_t   patch_sz_     ;

private:
    void do_initialize_work()
    {
        // check rep
        ZI_ASSERT((!F_in_)&&(!F_in_fft_)&&(!W_fft_));
        ZI_ASSERT(initialize_task_==nullptr);
        ZI_ASSERT(in_size_!=vec3i::zero);

        if (is_fft_)
        {
            // if ( sparse_ != vec3i::one )
            //     W_fft_ = fftw::forward_sparse_pad(W_, sparse_, in_size_);
            // else
            //     W_fft_ = fftw::forward_pad(W_, in_size_);
            // momentum volume logic
        }
    }

public:

    void forward( const vol_p<double>& F )
    {
        ZI_ASSERT(!is_fft_);

        bool should_init = false;
        bool should_fwd  = false;

        {
            std::unique_lock<std::mutex> guard(mutex_);

            switch (state_)
            {
            case state::ready:
                // out_->forward(F,W_);
                break;

            case state::uninitialized:
                ZI_ASSERT(initialize_task_);
                initialize_task_ = nullptr;
                break;


            default:
                DIE("In illegal state on forward");
            }
        }
    }

}; // class node_base

}} // namespace zi::znn


#endif //  ZNN_CORE_EDGE_HPP_INCLUDED
