#pragma once

#include "../../utils/max_accumulator.hpp"
#include "edges_fwd.hpp"
#include "nodes.hpp"

namespace znn { namespace v4 { namespace parallel_network {

class maxout_edge: public edge
{
public:

    // individual maxout group
    class maxout
    {
    private:
        max_accumulator accum_ ;
        cube_p<int>     indices;

    private:
        cube_p<real> forward(cube_p<real>&& f, size_t idx)
        {
            cube_p<real> ret = nullptr;

            if ( accum_.add(std::move(f), idx) )
            {
                auto r  = accum_.reset();
                ret     = r.first;
                indices = r.second;
            }

            return ret;
        }

        cube_p<real> backward(ccube<real> const & g, size_t idx)
        {
            auto         r = get_cube<real>(size(*g));
            real*       rp = r->data();
            const real* gp = g->data();
            const int*  ip = indices->data();

            for ( size_t i = 0; i < rp.num_elements(); ++i )
            {
                if ( ip[i] == idx )
                    rp[i] = gp[i];
                else
                    rp[i] = 0;
            }

            return r;
        }

    public:
        maxout( size_t n )
            : accum_(n)
        {}
    };

    // a layer of maxout groups
    class layer
    {
    private:
        std::vector<maxout> maxouts_;
        size_t group;
        size_t count = 0;

        friend class maxout_edge;

    public:
        layer( size_t n, size_t g )
            : group(g)
        {
            for ( size_t i = 0; i < n; ++i )
                maxouts_.emplace_back(g);
        }

        size_t register()
        {
            ZI_ASSERT(count<group);
            return count++;
        }
    };

private:
    maxout * maxout_data;
    size_t group_idx;

public:
    maxout_edge( nodes * in,
                 size_t inn,
                 nodes * out,
                 size_t outn,
                 task_manager & tm,
                 layer const & layer,
                 size_t idx )
        : edge(in,inn,out,outn,tm)
        , maxout_data(&layer.maxouts_[outn])
        , group_idx(idx)
    {
        ZI_ASSERT(inn=outn);
        in->attach_out_edge(inn,this);
        out->attach_in_edge(outn,this);
    }

    void forward( ccube_p<real> const & f ) override
    {
        auto r = maxout_data->forward(get_copy(*f), group_idx);
        if ( r ) out_nodes->forward(out_num, r);
    }

    void backward( ccube_p<real> const & g ) override
    {
        if ( in_nodes->is_input() )
        {
            in_nodes->backward(in_num, cube_p<real>());
        }
        else
        {
            auto r = maxout_data->backward(*g, group_idx);
            in_nodes->backward(in_num, r);
        }
    }

    void zap(edges* e)
    {
        e->edge_zapped();
    }
};

}}} // namespace znn::v4::parallel_network
