#ifndef ZNN_ERROR_FN_HPP_INCLUDED
#define ZNN_ERROR_FN_HPP_INCLUDED

#include "../core/types.hpp"

namespace zi {
namespace znn {

class error_fn
{
public:
    virtual double3d_ptr gradient(double3d_ptr /* dEdF */,
                                  double3d_ptr /* F    */) = 0;

    virtual void apply(double3d_ptr) = 0;

    virtual void add_apply(double c, double3d_ptr v)
    {
        std::size_t n  = v->shape()[0]*v->shape()[1]*v->shape()[2];
        for ( std::size_t i = 0; i < n; ++i )
        {
            v->data()[i] += c;
        }
        this->apply(v);
    }

}; // abstract class error_fn

typedef boost::shared_ptr<error_fn> error_fn_ptr;

}} // namespace zi::znn

#endif // ZNN_ERROR_FN_HPP_INCLUDED
