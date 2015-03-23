//
// Copyright (C) 2015-present  Aleksandar Zlateski <zlateski@mit.edu>
// ----------------------------------------------------------
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

#ifndef ZNN_CORE_BIAS_HPP_INCLUDED
#define ZNN_CORE_BIAS_HPP_INCLUDED

namespace zi {
namespace znn {

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

}} // namespace zi::znn

#endif // ZNN_CORE_BIAS_HPP_INCLUDED
