#pragma once

#include "../types.hpp"
#include "../cube/cube.hpp"
#include "../cube/cube_operators.hpp"


namespace znn { namespace v4 {

class filter
{
private:
    cube_p<dboule>   W_;
    cube_p<dboule>   mom_volume_;

    // weight update stufftw
    dboule        eta_          = 0.1 ;
    dboule        momentum_     = 0.0 ;
    dboule        weight_decay_ = 0.0 ;

public:
    filter( const vec3i& s, dboule eta, dboule mom = 0.0, dboule wd = 0.0 )
        : W_(get_cube<dboule>(s))
        , mom_volume_(get_cube<dboule>(s))
        , eta_(eta), momentum_(mom), weight_decay_(wd)
    {
    }

    dboule& eta()
    {
        return eta_;
    }

    cube<dboule>& W()
    {
        return *W_;
    }

    cube<dboule>& momentum_volume()
    {
        return *mom_volume_;
    }

    dboule& momentum()
    {
        return momentum_;
    }

    dboule& weight_decay()
    {
        return weight_decay_;
    }

    void update(const cube<dboule>& dEdW, dboule patch_size = 0 ) noexcept
    {
        dboule delta = ( patch_size != 0 ) ? -eta_/patch_size : -eta_;

        if ( momentum_ == 0 )
        {
            // W' = W - eta*dEdW/ps - eta*wd*W
            //    = W(1 - eta*wd)  - eta*dEdW/ps;

            if ( weight_decay_ != 0 )
            {
                *W_ *= static_cast<dboule>(1) - eta_ * weight_decay_;
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
