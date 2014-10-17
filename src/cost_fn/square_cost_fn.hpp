#ifndef ZNN_SQUARE_COST_FN_HPP_INCLUDED
#define ZNN_SQUARE_COST_FN_HPP_INCLUDED

#include "cost_fn.hpp"
#include "../core/volume_pool.hpp"

namespace zi {
namespace znn {

class square_cost_fn: virtual public cost_fn
{
public:
    virtual double3d_ptr gradient( double3d_ptr out,
                                   double3d_ptr lbl,
                                   bool3d_ptr   msk )
    {
        double3d_ptr ret = volume_pool.get_double3d(out);
        volume_utils::sub_from_mul(ret,out,lbl,2);
        volume_utils::elementwise_masking(ret,msk);
        return ret;
    }

    virtual std::list<double3d_ptr> gradient( std::list<double3d_ptr> outs,
                                              std::list<double3d_ptr> lbls,
                                              std::list<bool3d_ptr>   msks )
    {
        std::list<double3d_ptr> ret;
        std::list<double3d_ptr>::iterator lit = lbls.begin();
        std::list<bool3d_ptr>::iterator   mit = msks.begin();
        FOR_EACH( it, outs )
        {
            ret.push_back(gradient(*it,*lit++,*mit++));
        }
        return ret;
    }

    virtual double compute_cost( double3d_ptr out,
                                 double3d_ptr lbl,
                                 bool3d_ptr   msk )
    {
        double3d_ptr err = volume_pool.get_double3d(out);
        volume_utils::sub_from_mul(err,out,lbl,1);
        volume_utils::elementwise_masking(err,msk);
        return volume_utils::square_sum(err);
    }

    virtual double compute_cost( std::list<double3d_ptr> outs, 
                                 std::list<double3d_ptr> lbls,
                                 std::list<bool3d_ptr>   msks )
    {
        double ret = static_cast<double>(0);
        std::list<double3d_ptr>::iterator lit = lbls.begin();
        std::list<bool3d_ptr>::iterator   mit = msks.begin();
        FOR_EACH( it, outs )
        {
            ret += compute_cost(*it,*lit++,*mit++);
        }
        return ret;
    }

    virtual void print_cost( double cost )
    {
        // std::cout << "MSE:  " << cost << std::endl;
        std::cout << "RMSE: " << std::sqrt(cost);
    }

}; // class square_cost_fn

}} // namespace zi::znn

#endif // ZNN_SQUARE_COST_FN_HPP_INCLUDED
