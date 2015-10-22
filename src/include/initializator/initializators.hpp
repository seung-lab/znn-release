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

#include "zero_init.hpp"
#include "uniform_init.hpp"
#include "gaussian_init.hpp"
#include "normalize_init.hpp"
#include "constant_init.hpp"

#include "../assert.hpp"
#include "../options/options.hpp"

namespace znn { namespace v4 {

std::shared_ptr<initializator> get_initializator( options const & op )
{
    std::string fn = op.require_as<std::string>("init");

    if ( fn == "zero" )
    {
        return std::make_shared<zero_init>();
    }
    else if ( fn == "uniform" )
    {
        ovector<real> p
            = op.optional_as<ovector<real>>("init_args", "-0.1,0.1");

        ZI_ASSERT(p.size()&&p.size()<3);

        if ( p.size() == 1 )
        {
            return std::make_shared<uniform_init>(p[0]);
        }
        else
        {
            return std::make_shared<uniform_init>(p[0],p[1]);
        }
    }
    else if ( fn == "constant" )
    {
        real v = op.require_as<real>("init_args");
        return std::make_shared<constant_init>(v);
    }
    else if ( fn == "gaussian" )
    {
        ovector<real> p
            = op.optional_as<ovector<real>>("init_args", "0,0.01");

        ZI_ASSERT(p.size()==2);

        return std::make_shared<gaussian_init>(p[0],p[1]);
    }

    throw std::logic_error(HERE() + "unknown init function: " + fn);
}

}} // namespace znn::v4
