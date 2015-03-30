#pragma once

#include "../types.hpp"
#include "../cube/cube.hpp"
#include "../cube/cube_operators.hpp"


namespace znn { namespace v4 {

class filter
{
private:
    cube_p<double>   W_;
    cube_p<double>   mom_volume_;

    // weight update stufftw
    double        eta_          = 0.1 ;
    double        momentum_     = 0.0 ;
    double        weight_decay_ = 0.0 ;

public:
    filter( const vec3i& s, double eta, double mom = 0.0, double wd = 0.0 )
        : W_(get_cube<double>(s))
        , mom_volume_(get_cube<double>(s))
        , eta_(eta), momentum_(mom), weight_decay_(wd)
    {
    }

    double& eta()
    {
        return eta_;
    }

    cube<double>& W()
    {
        return *W_;
    }

    cube<double>& momentum_volume()
    {
        return *mom_volume_;
    }

    double& momentum()
    {
        return momentum_;
    }

    double& weight_decay()
    {
        return weight_decay_;
    }

    void update(const cube<double>& dEdW, double patch_size = 0 ) noexcept
    {
        double delta = ( patch_size != 0 ) ? -eta_/patch_size : -eta_;

        if ( momentum_ == 0 )
        {
            // W' = W - eta*dEdW/ps - eta*wd*W
            //    = W(1 - eta*wd)  - eta*dEdW/ps;

            if ( weight_decay_ != 0 )
            {
                *W_ *= static_cast<double>(1) - eta_ * weight_decay_;
            }

            mad_to( delta, dEdW, *W_ );
        }
        else
        {
            *mom_volume_ *= momentum_;
            mad_to( delta, dEdW, *mom_volume_ );

            if ( weight_decay_ != 0 )
            {
                mad_to( -eta_ * weight_decay_, *W_, *mom_volume_ );
            }

            *W_ += *mom_volume_;
        }
    }
}; // class filter

}} // namespace znn::v4
