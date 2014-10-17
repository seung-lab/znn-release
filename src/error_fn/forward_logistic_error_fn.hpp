#ifndef ZNN_FORWARD_LOGISTIC_ERROR_FN_HPP_INCLUDED
#define ZNN_FORWARD_LOGISTIC_ERROR_FN_HPP_INCLUDED

#include "error_fn.hpp"
#include "../core/volume_pool.hpp"

namespace zi {
namespace znn {

class forward_logistic_error_fn: virtual public logistic_error_fn
{
public:
    virtual double3d_ptr gradient(double3d_ptr dEdF, double3d_ptr F)
    {
        double3d_ptr r = volume_pool.get_double3d(F);
        (*r) = (*dEdF);
        return r;
    }

}; // class forward_logistic_error_fn

}} // namespace zi::znn

#endif // ZNN_FORWARD_LOGISTIC_ERROR_FN_HPP_INCLUDED
