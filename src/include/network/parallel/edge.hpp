#pragma once

#include "../../assert.hpp"
#include "../../types.hpp"
#include "../../cube/cube.hpp"
#include "../../utils/task_manager.hpp"
#include "nodes.hpp"

namespace znn { namespace v4 { namespace parallel_network {

class edges;

class edge
{
protected:
    nodes * in_nodes ;
    size_t  in_num   ;
    nodes * out_nodes;
    size_t  out_num  ;

    task_manager & manager;

    size_t fwd_priority_;
    size_t bwd_priority_;

public:
    edge( nodes * in, size_t inn, nodes * out, size_t outn, task_manager & m )
        : in_nodes(in), in_num(inn), out_nodes(out), out_num(outn), manager(m)
    {
        fwd_priority_ = out->fwd_priority() * 1024 + outn;
        bwd_priority_ = in->bwd_priority() * 1024 + inn;
        //fwd_priority_ = outn;
        //bwd_priority_ = inn;
    }

    size_t fwd_priority() const { return fwd_priority_; }
    size_t bwd_priority() const { return bwd_priority_; }

    virtual ~edge() {}

    virtual void forward( ccube_p<real> const & )
    { UNIMPLEMENTED(); }

    virtual void backward( ccube_p<real> const & )
    { UNIMPLEMENTED(); }

    virtual void forward( ccube_p<complex> const & )
    { UNIMPLEMENTED(); }

    virtual void backward( ccube_p<complex> const & )
    { UNIMPLEMENTED(); }

};



}}} // namespace znn::v4::parallel_network
