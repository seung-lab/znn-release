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

#ifndef ZNN_CORE_INITIALIZATOR_INITIALIZATOR_HPP_INCLUDED
#define ZNN_CORE_INITIALIZATOR_INITIALIZATOR_HPP_INCLUDED

#include <random>
#include <limits>
#include <mutex>
#include <ctime>
#include <zi/utility/singleton.hpp>
#include "../types.hpp"
#include "../volume_operators.hpp"

namespace zi {
namespace znn {

namespace detail {

struct random_number_generator_impl: std::mutex
{
    std::mt19937 rng = std::mt19937(std::time(0));
    //     std::mt19937 rng = std::mt19937(std::random_device(0));
}; // class random_number_generator_impl

} // namespace detail

class zinitializator
{
protected:
    template <typename T>
    static void initialize_with_distribution(T& dis, vol<double>& v) noexcept
    {
        static detail::random_number_generator_impl& rng =
            zi::singleton<detail::random_number_generator_impl>::instance();

        mutex_guard g(rng);

        for ( std::size_t i = 0; i < v.num_elements(); ++i )
        {
            v.data()[i] = dis(rng.rng);
        }
    }

    virtual void do_initialize( vol<double>& ) noexcept = 0;

public:

    void initialize( vol<double>& v )         { this->do_initialize(v);  }
    void initialize( const vol_p<double>& v ) { this->do_initialize(*v); }
}; // class initializator

}} // namespace zi::znn


#endif // ZNN_CORE_INITIALIZATOR_INITIALIZATOR_HPP_INCLUDED
