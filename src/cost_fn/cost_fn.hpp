#ifndef ZNN_COST_FN_HPP_INCLUDED
#define ZNN_COST_FN_HPP_INCLUDED

#include "../core/types.hpp"
#include "../core/volume_utils.hpp"

namespace zi {
namespace znn {

class cost_fn
{
public:
    virtual std::list<double3d_ptr> gradient( std::list<double3d_ptr> outputs,
                                              std::list<double3d_ptr> labels,
                                              std::list<bool3d_ptr>   masks ) = 0;

    virtual double compute_cost( std::list<double3d_ptr> outputs,
                                 std::list<double3d_ptr> labels,
                                 std::list<bool3d_ptr>   masks ) = 0;

    virtual double compute_cls_error( std::list<double3d_ptr> outputs,
                                      std::list<double3d_ptr> labels,
                                      std::list<bool3d_ptr>   masks,
                                      double threshold = 0.5 )
    {
        double ret = static_cast<double>(0);
        std::list<double3d_ptr>::iterator lit = labels.begin();
        std::list<bool3d_ptr>::iterator mit = masks.begin();
        FOR_EACH( it, outputs )
        {
            double3d_ptr err = volume_utils::classification_error(*it, *lit++, threshold);
            volume_utils::elementwise_masking(err, *mit++);
            ret += volume_utils::sum_all(err);
        }
        return ret;
    }

    virtual double get_output_number( bool3d_ptr mask )
    {
        return volume_utils::nnz(mask);
    }

    virtual double get_output_number( std::list<bool3d_ptr> masks )
    {
        return volume_utils::nnz(masks);
    }

    virtual void print_cost( double cost ) = 0;

    virtual void print_cls_error( double cls_err )
    {
        std::cout << "CLSE: " << cls_err;
    }

}; // abstract class cost_fn

typedef boost::shared_ptr<cost_fn> cost_fn_ptr;

}} // namespace zi::znn

#endif // ZNN_COST_FN_HPP_INCLUDED