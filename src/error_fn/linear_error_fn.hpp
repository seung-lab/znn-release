#ifndef ZNN_LINEAR_ERROR_FN_HPP_INCLUDED
#define ZNN_LINEAR_ERROR_FN_HPP_INCLUDED

#include "error_fn.hpp"
#include "../core/volume_pool.hpp"

#include <algorithm>
#include <utility>

namespace zi {
namespace znn {

class linear_error_fn: virtual public error_fn
{
private:
    double a_;  // a*x + b
    double b_;  // a*x + b

public:
    virtual double3d_ptr gradient(double3d_ptr dEdF, double3d_ptr F)
    {
        std::size_t n = F->num_elements();        
        double3d_ptr r = volume_pool.get_double3d(F);
        for ( std::size_t i = 0; i < n; ++i )
        {
            r->data()[i] = dEdF->data()[i] * a_;
        }
        return r;
    }

    virtual void apply(double3d_ptr v)
    {
        std::size_t n = v->num_elements();
        for ( std::size_t i = 0; i < n; ++i )
        {
            v->data()[i] = a_*v->data()[i] + b_;
        }   
    }

    virtual void add_apply(double c, double3d_ptr v)
    {
        std::size_t n = v->num_elements();
        for ( std::size_t i = 0; i < n; ++i )
        {
            v->data()[i] = a_*(c + v->data()[i]) + b_;
        }
    }


public:
    linear_error_fn(double a = static_cast<double>(1), 
                    double b = static_cast<double>(0))
        : a_(a)
        , b_(b)
    {}

}; // class linear_error_fn

}} // namespace zi::znn

#endif // ZNN_LINEAR_ERROR_FN_HPP_INCLUDED
