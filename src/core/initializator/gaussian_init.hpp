//
// Copyright (C) 2015-present  Aleksandar Zlateski <zlateski@mit.edu>
//                             Kisuk Lee           <kisuklee@mit.edu>
// ------------------------------------------------------------------
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

#ifndef ZNN_CORE_INITIALIZATOR_GAUSSIAN_INIT_HPP_INCLUDED
#define ZNN_CORE_INITIALIZATOR_GAUSSIAN_INIT_HPP_INCLUDED

#include "initializator.hpp"

namespace zi {
namespace znn {

class gaussian_init: public zinitializator
{
private:
    std::normal_distribution<double> dis;

    void do_initialize( vol<double>& v ) noexcept override
    {
        zinitializator::initialize_with_distribution(dis, v);
    }

public:
    explicit gaussian_init( double mu = 0, double sigma = 0.01 )
        : dis(mu, sigma)
    {}

}; // class gaussian_init

}} // namespace zi::znn

#endif // ZNN_CORE_INITIALIZATOR_GAUSSIAN_INIT_HPP_INCLUDED
