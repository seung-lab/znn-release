#pragma once

#include "../../../types.hpp"

#include <zi/utility/singleton.hpp>
#include <mkl_vsl.h>
#include <unordered_map>

namespace znn { namespace v4 { namespace detail {

class single_size_conv_plans
{
private:
    std::mutex                      m1       ;
    std::mutex                      m2       ;
    MKL_INT                         shape[3] ;

    std::unordered_map<vec3i, VSLConvTaskPtr, vec_hash<vec3i>> plans1   ;
    std::unordered_map<vec3i, VSLConvTaskPtr, vec_hash<vec3i>> plans2   ;

public:
    single_size_conv_plans( vec3i const & s )
    {
        shape[2] = s[0];
        shape[1] = s[1];
        shape[0] = s[2];
    }

    VSLConvTaskPtr get( vec3i const & s )
    {
        VSLConvTaskPtr& task = plans1[s];
        if ( task ) return task;

        int status;
        const int start[3]={s[2]-1,s[1]-1,s[0]-1};

        MKL_INT bshape[3] = { s[2], s[1], s[0] };
        MKL_INT rshape[3] = { shape[0] + 1 - s[2],
                              shape[1] + 1 - s[1],
                              shape[2] + 1 - s[0] };

#ifdef ZNN_USE_FLOATS
        status = vslsConvNewTask(&task,VSL_CONV_MODE_DIRECT,3,
                                 shape, bshape, rshape);
#else
        status = vsldConvNewTask(&task,VSL_CONV_MODE_DIRECT,3,
                                 shape, bshape, rshape);
#endif
        status = vslConvSetStart(task, start);
        return task;
    }

    VSLConvTaskPtr get_inv( vec3i const & s )
    {
        VSLConvTaskPtr& task = plans2[s];
        if ( task ) return task;

        int status;
        const int start[3]={0,0,0};

        MKL_INT bshape[3] = { s[2], s[1], s[0] };
        MKL_INT rshape[3] = { shape[0] + s[2] - 1,
                              shape[1] + s[1] - 1,
                              shape[2] + s[0] - 1};

#ifdef ZNN_USE_FLOATS
        status = vslsConvNewTask(&task,VSL_CONV_MODE_DIRECT,3,
                                 shape, bshape, rshape);
#else
        status = vsldConvNewTask(&task,VSL_CONV_MODE_DIRECT,3,
                                 shape, bshape, rshape);
#endif
        status = vslConvSetStart(task, start);

        return task;
    }


    VSLConvTaskPtr get_synchronized( vec3i const & s )
    {
        guard g(m1);
        return get(s);
    }

    VSLConvTaskPtr get_inv_synchronized( vec3i const & s )
    {
        guard g(m2);
        return get_inv(s);
    }

};

class conv_plans_impl
{
private:
    std::mutex                                                          m    ;
    std::unordered_map<vec3i, single_size_conv_plans*, vec_hash<vec3i>> pools;

    single_size_conv_plans* get_pool(vec3i const & s)
    {
        typedef single_size_conv_plans* single_size_conv_plans_ptr;
        guard g(m);
        single_size_conv_plans_ptr& r = pools[s];
        if ( r ) return r;
        r = new single_size_conv_plans(s);
        return r;
    }

    bool locked = true;

public:
    VSLConvTaskPtr get(vec3i const & a, vec3i const & b)
    {
        if (locked)
            return get_pool(a)->get_synchronized(b);
        else
            return pools[a]->get(b);
    }

    VSLConvTaskPtr get_inv(vec3i const & a, vec3i const & b)
    {
        if (locked)
            return get_pool(a)->get_inv_synchronized(b);
        else
            return pools[a]->get_inv(b);
    }

    void lock()
    {
        locked = true;
    }

    void unlock()
    {
        locked = false;
    }
};

namespace {
conv_plans_impl& conv_plans = zi::singleton<conv_plans_impl>::instance();
} // anonymous namespace


}}} // namespace znn::v4::convolution
