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
    vec3i const    fsize_       ;
    task_manager & task_manager_;

protected:
    nodes( vec3i const & fsize, task_manager & tm )
        : fsize_(fsize), task_manager_(tm)
    {
    }

public:
    vec3i const &  fsize() const { return fsize_;        }
    task_manager & manager()     { return task_manager_; }

public:
    virtual ~nodes() {}

    // receive a featuremap for the i-th input
    // featuremap is absorbed
    virtual void forward(size_t, cube_p<double>&&)
    { UNIMPLEMENTED(); }

    // receive a gradient for the i-th output
    // gradient is absorbed
    virtual void backward(size_t, cube_p<double>&&)
    { UNIMPLEMENTED(); }

    // for inplace convolution
    virtual void forward(size_t,
                         ccube_p<double> const & /* featuremap */,
                         ccube_p<double> const & /* filter */,
                         vec3i const &           /* filter_stride */ )
    { UNIMPLEMENTED(); }

    // for inplace convolution
    virtual void backward(size_t,
                          ccube_p<double> const & /* gradient */,
                          ccube_p<double> const & /* filter */,
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


    virtual std::vector<cube_p<double>>& get_featuremaps()
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

    virtual void set_eta( double )
    { UNIMPLEMENTED(); }

    virtual void set_momentum( double )
    { UNIMPLEMENTED(); }

    virtual void set_weight_decay( double )
    { UNIMPLEMENTED(); }

    virtual bool is_input() const
    { return false; }

    virtual bool is_output() const
    { return false; }

    virtual void zap() = 0;
    virtual options serialize() const = 0;

};


}}} // namespace znn::v4::parallel_network
