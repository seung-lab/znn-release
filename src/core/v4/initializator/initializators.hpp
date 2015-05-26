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
        ovector<dboule> p
            = op.optional_as<ovector<dboule>>("init_args", "-0.1,0.1");

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
        dboule v = op.require_as<dboule>("init_args");
        return std::make_shared<constant_init>(v);
    }
    else if ( fn == "gaussian" )
    {
        ovector<dboule> p
            = op.optional_as<ovector<dboule>>("init_args", "0,0.01");

        ZI_ASSERT(p.size()==2);

        return std::make_shared<gaussian_init>(p[0],p[1]);
    }

    throw std::logic_error(HERE() + "unknown init function: " + fn);
}

}} // namespace znn::v4
