#pragma once

#include "../../assert.hpp"
#include "../../types.hpp"
#include "../../cube/cube.hpp"

namespace znn { namespace v4 { namespace parallel_network {

class edge
{
public:
    virtual ~edge() {}

    virtual void forward( ccube_p<double> const & )
    { UNIMPLEMENTED(); }

    virtual void backward( ccube_p<double> const & )
    { UNIMPLEMENTED(); }

    virtual void forward( ccube_p<complex> const & )
    { UNIMPLEMENTED(); }

    virtual void backward( ccube_p<complex> const & )
    { UNIMPLEMENTED(); }
};

}}} // namespace znn::v4::parallel_network
