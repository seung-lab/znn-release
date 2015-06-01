#pragma once

#include "../types.hpp"
#include "../cube/cube.hpp"
#include "../cube/cube_operators.hpp"


namespace znn { namespace v4 {

class filter
{
private:
    cube_p<real>   W_;
    cube_p<real>   mom_volume_;

    // weight update stufftw
    real        eta_          = 0.1 ;
    real        momentum_     = 0.0 ;
    real        weight_decay_ = 0.0 ;

public:
    filter( const vec3i& s, real eta, real mom = 0.0, real wd = 0.0 )
        : W_(get_cube<real>(s))
        , mom_volume_(get_cube<real>(s))
        , eta_(eta), momentum_(mom), weight_decay_(wd)
    {
    }

    real& eta()
    {
        return eta_;
    }

    cube<real>& W()
    {
        return *W_;
    }

    cube<real>& momentum_volume()
    {
        return *mom_volume_;
    }

    real& momentum()
    {
        return momentum_;
    }

    real& weight_decay()
    {
        return weight_decay_;
    }

    void update(const cube<real>& dEdW, real patch_size = 0 ) noexcept
    {
        real delta = ( patch_size != 0 ) ? -eta_/patch_size : -eta_;

        if ( momentum_ == 0 )
        {
            // W' = W - eta*dEdW/ps - eta*wd*W
            //    = W(1 - eta*wd)  - eta*dEdW/ps;

            if ( weight_decay_ != 0 )
            {
                *W_ *= static_cast<real>(1) - eta_ * weight_decay_;
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
