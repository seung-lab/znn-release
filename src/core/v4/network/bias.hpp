#pragma once

namespace znn { namespace v4 {

class bias
{
private:
    double   b_;
    double   v_;

    // weight update stuff
    double   eta_  = 0.1 ;
    double   mom_  = 0.0 ;
    double   wd_   = 0.0 ;

public:
    bias( double eta, double mom = 0.0, double wd = 0.0 )
        : b_(0), v_(0), eta_(eta), mom_(mom), wd_(wd)
    {
    }

    double& eta()
    {
        return eta_;
    }

    double& b()
    {
        return b_;
    }

    double& momentum_value()
    {
        return v_;
    }

    double& momentum()
    {
        return mom_;
    }

    double& weight_decay()
    {
        return wd_;
    }

    void update(double dEdB, double patch_size = 1 ) noexcept
    {
        v_ = (mom_*v_) - (eta_*wd_*b_) - (eta_*dEdB/patch_size);
        b_ += v_;
    }

}; // class bias

}} // namespace znn::v4
