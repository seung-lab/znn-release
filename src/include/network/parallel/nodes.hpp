#pragma once

#include "../../assert.hpp"
#include "../../options/options.hpp"
#include "../../utils/task_manager.hpp"
#include "../../cube/cube.hpp"

namespace znn { namespace v4 { namespace parallel_network {

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

protected:
    nodes( size_t sz,
           vec3i const & fsize,
           options const & op,
           task_manager & tm,
           bool is_in = false,
           bool is_out = false )
        : size_(sz)
        , fsize_(fsize)
        , task_manager_(tm)
        , options_(op)
        , is_input_(is_in)
        , is_output_(is_out)
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
