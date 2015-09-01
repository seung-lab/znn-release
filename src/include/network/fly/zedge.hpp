#pragma once

namespace znn { namespace v4 { namespace fly {

class node;

class edge
{
    size_t fwd_priority() const { return 0; }
    size_t bwd_priority() const { return 0; }

    virtual void forward( ccube_p<real> const & )
    { UNIMPLEMENTED(); }

    virtual void backward( ccube_p<real> const & )
    { UNIMPLEMENTED(); }

    virtual void forward( ccube_p<complex> const & )
    { UNIMPLEMENTED(); }

    virtual void backward( ccube_p<complex> const & )
    { UNIMPLEMENTED(); }

    virtual void zap(edges*) = 0;
}

template<class Implementation>
class edge
{

};

}}} // namespace znn::v4::fly
