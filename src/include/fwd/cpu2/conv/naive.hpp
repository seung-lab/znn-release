#pragma once

#include "../../assert.hpp"
#include "../../types.hpp"

namespace znn { namespace fwd {

class convolver
{
private:
    class ref
    {
    private:
        long_t xs, ys;
        real*  data;

    public:
        ref( long_t y, long_t z, real* d )
            : xs(y*z)
            , ys(z)
            , data(d)
        {}

        real & operator()( long_t x, long_t y, long_t z )
        {
            return data[x*xs + y*ys + z];
        }
    };

private:
    vec3i ix, kx, rx;

public:
    convolver( vec3i const & i, vec3i const & k )
        : ix(i)
        , kx(k)
        , rx(i - k + vec3i::one)
    { }

    void convolve( real* in, real* kernel, real* out )
    {
        ref a(ix[1], ix[2], in);
        ref b(kx[1], kx[2], kernel);
        ref r(rx[1], rx[2], out);

        for ( long_t x = 0; x < rx[0]; ++x )
            for ( long_t y = 0; y < rx[1]; ++y )
                for ( long_t z = 0; z < rx[2]; ++z )
                {
                    real & res = r(x,y,z);
                    res = 0;
                    for ( long_t dx = x, wx = kx[0] - 1; dx < kx[0] + x; ++dx, --wx )
                        for ( long_t dy = y, wy = kx[1] - 1; dy < kx[1] + y; ++dy, --wy )
                            for ( long_t dz = z, wz = kx[2] - 1; dz < kx[2] + z; ++dz, --wz )
                                res += a(dx,dy,dz) * b(wx,wy,wz);
                }
    }

    void convolve_add( real* in, real* kernel, real* out )
    {
        ref a(ix[1], ix[2], in);
        ref b(kx[1], kx[2], kernel);
        ref r(rx[1], rx[2], out);

        for ( long_t x = 0; x < rx[0]; ++x )
            for ( long_t y = 0; y < rx[1]; ++y )
                for ( long_t z = 0; z < rx[2]; ++z )
                {
                    real & res = r(x,y,z);
                    for ( long_t dx = x, wx = kx[0] - 1; dx < kx[0] + x; ++dx, --wx )
                        for ( long_t dy = y, wy = kx[1] - 1; dy < kx[1] + y; ++dy, --wy )
                            for ( long_t dz = z, wz = kx[2] - 1; dz < kx[2] + z; ++dz, --wz )
                                res += a(dx,dy,dz) * b(wx,wy,wz);
                }
    }
};


}} // namespace znn::fwd
