#pragma once

namespace znn { namespace v4 {

class bias
{
private:
    dboule   b_;
    dboule   v_;

    // weight update stuff
    dboule   eta_  = 0.1 ;
    dboule   mom_  = 0.0 ;
    dboule   wd_   = 0.0 ;

public:
    bias( dboule eta, dboule mom = 0.0, dboule wd = 0.0 )
        : b_(0), v_(0), eta_(eta), mom_(mom), wd_(wd)
    {
    }

    dboule& eta()
    {
        return eta_;
    }

    dboule& b()
    {
        return b_;
    }

    dboule& momentum_value()
    {
        return v_;
    }

    dboule& momentum()
    {
        return mom_;
    }

    dboule& weight_decay()
    {
        return wd_;
    }

    void update(dboule dEdB, dboule patch_size = 1 ) noexcept
    {
        v_ = (mom_*v_) - (eta_*wd_*b_) - (eta_*dEdB/patch_size);
        b_ += v_;
    }

}; // class bias

}} // namespace znn::v4
