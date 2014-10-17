#ifndef ZNN_BINOMIAL_CROSS_ENTROPY_COST_FN_HPP_INCLUDED
#define ZNN_BINOMIAL_CROSS_ENTROPY_COST_FN_HPP_INCLUDED

#include "cost_fn.hpp"
#include "../core/volume_pool.hpp"

#include <iomanip>

namespace zi {
namespace znn {

class binomial_cross_entropy_cost_fn: virtual public cost_fn
{
public:
    virtual std::list<double3d_ptr> gradient( std::list<double3d_ptr> probs,
                                              std::list<double3d_ptr> labels,
                                              std::list<bool3d_ptr>   masks )
    {
        std::list<double3d_ptr> ret;
        std::list<double3d_ptr>::iterator lit = labels.begin();
        std::list<bool3d_ptr>::iterator   mit = masks.begin();
        FOR_EACH( it, probs )
        {
            double3d_ptr grad = volume_pool.get_double3d(*it);
            volume_utils::sub_from_mul(grad,*it,*lit++,1);
            volume_utils::elementwise_masking(grad, *mit++);
            ret.push_back(grad);
        }
        return ret;
    }

    virtual double compute_cost( std::list<double3d_ptr> probs,
                                 std::list<double3d_ptr> labels,
                                 std::list<bool3d_ptr>   masks )
    {
        std::list<double3d_ptr> cost = 
            volume_utils::binomial_cross_entropy(probs,labels);

        double ret = 0;

        std::list<bool3d_ptr>::iterator mit = masks.begin();
        FOR_EACH( it, cost )
        {
            volume_utils::elementwise_masking(*it,*mit++);
            ret += volume_utils::sum_all(*it);
        }
        
        return ret;
    }

    virtual double compute_cls_error( std::list<double3d_ptr> probs,
                                      std::list<double3d_ptr> labels,
                                      std::list<bool3d_ptr>   masks,
                                      double thresh = 0.5 )
    {
        double3d_ptr ret = volume_pool.get_double3d(probs.front());
        volume_utils::zero_out(ret);

        std::list<double3d_ptr>::iterator lit = labels.begin();
        std::list<bool3d_ptr>::iterator   mit = masks.begin();
        FOR_EACH( it, probs )
        {
            double3d_ptr err = volume_utils::classification_error(*it,*lit++,thresh);
            volume_utils::elementwise_masking(err,*mit++);
            volume_utils::add_to(err,ret);
        }        
        return volume_utils::sum_all(ret);
    }

    virtual void print_cost( double cost )
    {
        std::cout << "logprob: " << cost;
    }

}; // class binomial_cross_entropy_cost_fn

}} // namespace zi::znn

#endif // ZNN_BINOMIAL_CROSS_ENTROPY_COST_FN_HPP_INCLUDED
