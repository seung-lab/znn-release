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

#ifndef ZNN_CORE_NETWORK_EDGE_EDGE_OPTIONS_HPP
#define ZNN_CORE_NETWORK_EDGE_EDGE_OPTIONS_HPP

#include "../../meta.hpp"

namespace zi {
namespace znn {

template<class... Args>
struct contains_tag: std::false_type {};

template<class X, class... Args>
struct contains_tag<X,X,Args...>: std::true_type {};

template<class X, class Y, class... Args>
struct contains_tag<X,Y,Args...>: contains_tag<X,Args...> {};

struct works_on_ffts {};
struct async_forward {};
struct async_backward{};
struct is_convolving {};
struct async_zap     {};
struct is_dummy      {};

template<class... Args>
struct edge_options
{
    template<typename Tag>
    using contains = contains_tag<Tag,Args...>;
};


}} // namespace zi::znn


#endif // ZNN_CORE_NETWORK_EDGE_EDGE_OPTIONS_HPP
