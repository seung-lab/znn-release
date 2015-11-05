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

#include "../../assert.hpp"
#include "../../options/options.hpp"
#include "../../utils/task_manager.hpp"
#include "../../cube/cube.hpp"

namespace znn { namespace v4 { namespace parallel_network {

enum class phase : std::uint8_t {TRAIN = 0, TEST = 1};

// Forward definition
class edge;

class nodes
{
private:
    size_t const   size_        ;
    vec3i  const   fsize_       ;
    task_manager & task_manager_;
    options        options_     ;
    bool           is_input_    ;
    bool           is_output_   ;

    size_t const   fwd_priority_;
    size_t const   bwd_priority_;

protected:
    real           patch_sz_ = 1; // minibatch averaging

    std::vector<bool>   enabled_;

protected:
    nodes( size_t sz,
           vec3i const & fsize,
           options const & op,
           task_manager & tm,
           size_t fwd_p,
           size_t bwd_p,
           bool is_in = false,
           bool is_out = false )
        : size_(sz)
        , fsize_(fsize)
        , task_manager_(tm)
        , options_(op)
        , is_input_(is_in)
        , is_output_(is_out)
        , fwd_priority_(fwd_p)
        , bwd_priority_(bwd_p)
        , enabled_(sz,true)
    {
    }

    options & opts() { return options_; }
    options const & opts() const { return options_; }

public:
    bool is_input()  { return is_input_ ; }
    bool is_output() { return is_output_; }

    vec3i const &  fsize() const { return fsize_;        }
    task_manager & manager()     { return task_manager_; }
    size_t         size() const  { return size_;         }

    size_t         fwd_priority() const { return fwd_priority_; }
    size_t         bwd_priority() const { return bwd_priority_; }

    void set_patch_size( real s )
    {
        ZI_ASSERT(s > 0);
        patch_sz_ = s;
    }

    std::string name() const
    {
        return options_.require_as<std::string>("name");
    }

public:
    virtual ~nodes() {}

    // receive a featuremap for the i-th input
    // featuremap is absorbed
    virtual void forward(size_t, cube_p<real>&&)
    { UNIMPLEMENTED(); }

    // receive a gradient for the i-th output
    // gradient is absorbed
    virtual void backward(size_t, cube_p<real>&&)
    { UNIMPLEMENTED(); }

    // for inplace convolution
    virtual void forward(size_t,
                         ccube_p<real> const & /* featuremap */,
                         ccube_p<real> const & /* filter */,
                         vec3i const &           /* filter_stride */ )
    { UNIMPLEMENTED(); }

    // for inplace convolution
    virtual void backward(size_t,
                          ccube_p<real> const & /* gradient */,
                          ccube_p<real> const & /* filter */,
                          vec3i const &           /* filter_stride */ )
    { UNIMPLEMENTED(); }

    // receive a featuremap for the i-th input
    // featuremap is absorbed
    virtual void forward(size_t, size_t, cube_p<complex>&&)
    { UNIMPLEMENTED(); }

    // receive a gradient for the i-th output
    // gradient is absorbed
    virtual void backward(size_t, size_t, cube_p<complex>&&)
    { UNIMPLEMENTED(); }

    // inplace multiply add
    virtual void forward(size_t, size_t,
                         ccube_p<complex> const & /* fft(featuremap) */,
                         ccube_p<complex> const & /* fft(filter) */ )
    { UNIMPLEMENTED(); }

    // inplace multiply add
    virtual void backward(size_t, size_t,
                          ccube_p<complex> const & /* fft(gradient) */,
                          ccube_p<complex> const & /* fft(filter) */ )
    { UNIMPLEMENTED(); }

    virtual std::vector<cube_p<real>>& get_featuremaps()
    { UNIMPLEMENTED(); }

    virtual size_t num_out_nodes()
    { UNIMPLEMENTED(); }

    virtual size_t num_in_nodes()
    { UNIMPLEMENTED(); }

    virtual void attach_out_edge(size_t, edge*)
    { UNIMPLEMENTED(); }

    virtual void attach_in_edge(size_t, edge*)
    { UNIMPLEMENTED(); }

    virtual size_t attach_out_fft_edge(size_t, edge*)
    { UNIMPLEMENTED(); }

    virtual size_t attach_in_fft_edge(size_t, edge*, vec3i const &)
    { UNIMPLEMENTED(); }

    virtual void enable(bool b)
    {
        for ( size_t i = 0; i < nodes::size(); ++i )
            enable(i,b);
    }

    virtual void enable(size_t, bool)
    { UNIMPLEMENTED(); }

    virtual void disable_out_edge(size_t)
    { UNIMPLEMENTED(); }

    virtual void disable_in_edge(size_t)
    { UNIMPLEMENTED(); }

    virtual void disable_out_fft_edge(size_t)
    { UNIMPLEMENTED(); }

    virtual void disable_in_fft_edge(size_t, vec3i const &)
    { UNIMPLEMENTED(); }

    virtual void set_eta( real )
    { UNIMPLEMENTED(); }

    virtual void set_momentum( real )
    { UNIMPLEMENTED(); }

    virtual void set_weight_decay( real )
    { UNIMPLEMENTED(); }

    virtual void wait()
    { UNIMPLEMENTED(); }

    virtual void zap() = 0;
    virtual options serialize() const = 0;

};


}}} // namespace znn::v4::parallel_network
