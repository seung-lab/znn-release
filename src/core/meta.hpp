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

#ifndef ZNN_CORE_META_HPP_INCLUDED
#define ZNN_CORE_META_HPP_INCLUDED

#include <type_traits>

namespace zi { namespace znn {

template<typename T>
struct identity { typedef T type; };

template<bool B>
using bool_constant = std::integral_constant<bool,B>;

template<class...>
struct void_t_helper_struct { typedef void type; };

template<class... Ts>
using void_t = typename void_t_helper_struct<Ts...>::type;


}} // namespace zi::znn


#endif // ZNN_CORE_META_HPP_INCLUDED
