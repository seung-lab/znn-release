#pragma once

#include "zero_init.hpp"
#include "uniform_init.hpp"
#include "gaussian_init.hpp"
#include "normalize_init.hpp"
#include "constant_init.hpp"

#include "../assert.hpp"
#include "../options/options.hpp"

namespace znn { namespace v4 {

std::shared_ptr<initializator> get_initializator( options const & op, options * info = nullptr )
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
    else if ( fn == "xavier")
    {
        // Initialization based on the paper [Bengio and Glorot 2010]
        // "Understanding the difficulty of training deep feedforward neuralnetworks"

        ZI_ASSERT(info);
        
        real n = info->require_as<real>("fan-in");
        real m = info->require_as<real>("fan-out");
        real r = std::sqrt(6)/std::sqrt(n + m);
        return std::make_shared<uniform_init>(-r,r);
    }
    else if ( fn == "msra")
    {
        // Initialization based on the paper [He, Zhang, Ren and Sun 2015]
        // Specifically accounts for ReLU nonlinearities.

        ZI_ASSERT(info);

        real n = info->require_as<real>("fan-in");
        real s = std::sqrt(2/n);
        return std::make_shared<gaussian_init>(0,s);
    }

    throw std::logic_error(HERE() + "unknown init function: " + fn);
}

}} // namespace znn::v4
