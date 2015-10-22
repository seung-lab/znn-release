//
// Copyright (C) 2012-2015  Aleksandar Zlateski <zlateski@mit.edu>
// ---------------------------------------------------------------
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.
//
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
