#pragma once

#include "../../assert.hpp"
#include "../../types.hpp"
#include "../../cube/cube.hpp"
#include "../../utils/task_manager.hpp"


namespace znn { namespace v4 { namespace parallel_network {

class nodes;
class edges;

class edge
{
protected:
    nodes * in_nodes ;
    size_t  in_num   ;
    nodes * out_nodes;
    size_t  out_num  ;

    task_manager & manager;

public:
    edge( nodes * in, size_t inn, nodes * out, size_t outn, task_manager & m )
        : in_nodes(in), in_num(inn), out_nodes(out), out_num(outn), manager(m)
    {
    }

    virtual ~edge() {}

    virtual void forward( ccube_p<real> const & )
    { UNIMPLEMENTED(); }

    virtual void backward( ccube_p<real> const & )
    { UNIMPLEMENTED(); }

    virtual void forward( ccube_p<complex> const & )
    { UNIMPLEMENTED(); }

    virtual void backward( ccube_p<complex> const & )
    { UNIMPLEMENTED(); }

    virtual void zap(edges*) = 0;
};



}}} // namespace znn::v4::parallel_network
