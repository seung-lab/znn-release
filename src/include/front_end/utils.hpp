//
// Copyright (C)      2016  Kisuk Lee           <kisuklee@mit.edu>
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

#include "../types.hpp"
#include "../options/types.hpp"
#include "../utils.hpp"

#include <algorithm>
#include <string>
#include <set>

namespace znn { namespace v4 {

template <typename T>
std::set<T> parse_number_set( std::string const & s )
{
    std::set<T> r;

    auto tokens = boost::lexical_cast<ovector<std::string>>(s);
    for ( auto& token: tokens )
    {
        try
        {
            r.insert(boost::lexical_cast<T>(token));
        }
        catch (...)
        {
            try
            {
                std::vector<std::string> parts;
                boost::split(parts, token, boost::is_any_of("-"));

                std::set<T> range;
                range.insert(boost::lexical_cast<T>(parts[0]));
                range.insert(boost::lexical_cast<T>(parts[1]));

                for ( T i = *range.begin(); i <= *range.rbegin(); ++i )
                    r.insert(i);
            }
            catch (...)
            {
                throw std::invalid_argument(token + " in " + s);
            }
        }

    }

    return r;
}

}} // namespace znn::v4