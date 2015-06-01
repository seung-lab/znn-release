#pragma once

#include "edge.hpp"
#include "nodes.hpp"
#include "../../utils/dispatcher.hpp"
#include "../../utils/waiter.hpp"

namespace znn { namespace v4 { namespace parallel_network {

class input_nodes: public nodes
{
private:
    dispatcher_group<concurrent_forward_dispatcher<edge,edge>> outputs_;
    waiter                                                     waiter_ ;

public:
    input_nodes( size_t s,
                 vec3i const & fsize,
                 options const & op,
                 task_manager & tm )
        : nodes(s,fsize,op,tm,true,false)
        , outputs_(s)
        , waiter_()
    {
    }

public:

    size_t num_out_nodes() override { return nodes::size(); }
    size_t num_in_nodes()  override { return nodes::size(); }

    void set_eta( real ) override {}
    void set_momentum( real ) override {}
    void set_weight_decay( real ) override {}

    options serialize() const { return nodes::opts(); }

    // For the forward pass only this function us supported
    void forward(size_t n, cube_p<real> && f) override
    {
        ZI_ASSERT(n<nodes::size());
        outputs_.dispatch(n,f,nodes::manager());
    }

private:
    void backward() { waiter_.one_done(); }

public:
    void backward(size_t, cube_p<real>&&) override
    { backward(); }

    void backward(size_t, ccube_p<real> const &,
                  ccube_p<real> const &, vec3i const &) override
    { backward(); }

    void backward(size_t, size_t, cube_p<complex>&&) override
    { backward(); }

    void backward(size_t, size_t,
                  ccube_p<complex> const &,
                  ccube_p<complex> const &) override
    { backward(); }

    void attach_out_edge(size_t i, edge* e) override
    {
        ZI_ASSERT(i<nodes::size());
        waiter_.inc();
        outputs_.sign_up(i,e);
    }

    size_t attach_out_fft_edge(size_t i, edge* e) override
    {
        ZI_ASSERT(i<nodes::size());
        waiter_.inc();
        outputs_.sign_up(i,nodes::fsize(),e);
        return 0;
    }

    void wait() override { waiter_.wait(); }

    void zap() override {}

}; // class input_nodes

}}} // namespace znn::v4::parallel_network
