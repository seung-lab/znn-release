#pragma once

namespace znn { namespace v4 {

class bias
{
private:
    real   b_;
    real   v_;

    // weight update stuff
    real   eta_  = 0.1 ;
    real   mom_  = 0.0 ;
    real   wd_   = 0.0 ;

public:
    bias( real eta, real mom = 0.0, real wd = 0.0 )
        : b_(0), v_(0), eta_(eta), mom_(mom), wd_(wd)
    {
    }

    real& eta()
    {
        return eta_;
    }

    real& b()
    {
        return b_;
    }

    real& momentum_value()
    {
        return v_;
    }

    real& momentum()
    {
        return mom_;
    }

    real& weight_decay()
    {
        return wd_;
    }

    void update(real dEdB, real patch_size = 1 ) noexcept
    {
        v_ = (mom_*v_) - (eta_*wd_*b_) - (eta_*dEdB/patch_size);
        b_ += v_;
    }

}; // class bias

}} // namespace znn::v4
