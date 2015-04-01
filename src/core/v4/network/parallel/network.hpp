#pragma once

#include "edges.hpp"
#include "input_nodes.hpp"
#include "transfer_nodes.hpp"

#include <map>

namespace znn { namespace v4 { namespace parallel_network {

class network
{
private:
    struct nnodes;

    struct nedges
    {
        vec3i width     = vec3i::one  ;
        vec3i stride    = vec3i::one  ;

        vec3i in_stride = vec3i::zero ;
        vec3i in_fsize  = vec3i::zero ;

        nnodes * in;
        nnodes * out;

        options const * opts;

        std::unique_ptr<edges> edges;
    };

    struct nnodes
    {
        vec3i fov       = vec3i::zero;
        vec3i stride    = vec3i::zero;
        vec3i fsize     = vec3i::zero;

        options const * opts;

        std::unique_ptr<nodes> nodes;
        std::vector<nedges *> in, out;
    };

private:
    network( network const & ) = delete;
    network& operator=( network const & ) = delete;

    network( network && ) = delete;
    network& operator=( network && ) = delete;

public:
    ~network()
    {
        for ( auto& n: nodes_ ) delete n.second;
        for ( auto& e: edges_ ) delete e.second;
    }

private:
    std::map<std::string, nedges*> edges_;
    std::map<std::string, nnodes*> nodes_;
    std::map<std::string, nnodes*> input_nodes_;
    std::map<std::string, nnodes*> output_nodes_;

    task_manager tm_;

    void fov_pass(nnodes* n, const vec3i& fov, const vec3i& fsize )
    {
        if ( n->fov != vec3i::zero )
        {
            ZI_ASSERT(n->fsize==fsize);
            ZI_ASSERT(n->fov==fov);
        }
        else
        {
            for ( auto& e: n->out )
            {
                e->in_fsize = fsize;
            }
            n->fov = fov;
            n->fsize = fsize;
            for ( auto& e: n->in )
            {
                vec3i new_fov   = (fov - vec3i::one) * e->stride + e->width;
                vec3i new_fsize = (e->width-vec3i::one) * e->in_stride + fsize;
                fov_pass(e->in, new_fov, new_fsize);
            }
        }
    }

    void stride_pass(nnodes* n, const vec3i& stride )
    {
        if ( n->stride != vec3i::zero )
        {
            ZI_ASSERT(n->stride==stride);
        }
        else
        {
            n->stride = stride;
            for ( auto& e: n->out )
            {
                e->in_stride = stride;
                stride_pass(e->out, stride * e->stride );
            }
        }
    }

    void init( vec3i const& outsz )
    {
        for ( auto& o: nodes_ )
            if ( o.second->out.size() == 0 )
                output_nodes_[o.first] = o.second;

        for ( auto& o: input_nodes_ )
            stride_pass(o.second, vec3i::one);
        for ( auto& o: output_nodes_ )
            fov_pass(o.second, vec3i::one, outsz);

        for ( auto& o: nodes_ )
        {
            std::cout << "NODE GROUP: " << o.first << "\n    "
                      << "FOV: " << o.second->fov << "\n    "
                      << "STRIDE: " << o.second->stride << "\n    "
                      << "SIZE: " << o.second->fsize << '\n';
        }

        // for ( auto& o: edges_ )
        // {
        //     std::cout << o.first << ' ' << o.second->width
        //               << ' ' << o.second->stride
        //               << ' ' << o.second->in_stride << '\n';
        // }

    }

    void add_nodes( options const & op )
    {
        auto name = op.require_as<std::string>("name");
        auto type = op.require_as<std::string>("type");

        ZI_ASSERT(nodes_.count(name)==0);
        nnodes* ns   = new nnodes;
        ns->opts = &op;
        nodes_[name] = ns;

        if ( type == "input" )
        {
            input_nodes_[name] = ns;
        }
    }

    void create_edges()
    {
        for ( auto & e: edges_ )
        {
            auto type = e.second->opts->require_as<std::string>("type");
            nodes * in  = e.second->in->nodes.get();
            nodes * out = e.second->out->nodes.get();

            if ( type == "max_filter" )
            {
                e.second->edges = std::make_unique<edges>
                    ( in, out, *e.second->opts, e.second->in_stride,
                      e.second->in_fsize, tm_, edges::max_pooling_tag() );
            }
            else if ( type == "conv" )
            {
                e.second->edges = std::make_unique<edges>
                    ( in, out, *e.second->opts, e.second->in_stride,
                      e.second->in_fsize, tm_, edges::filter_tag() );
            }
            else if ( type == "dummy" )
            {
                e.second->edges = std::make_unique<edges>
                    ( in, out, *e.second->opts, tm_, edges::dummy_tag() );
            }
            else
            {
                throw std::logic_error(HERE() + "unknown nodes type: " + type);
            }

            e.second->opts = nullptr;
        }
    }


    void create_nodes()
    {
        for ( auto & n: nodes_ )
        {
            auto type = n.second->opts->require_as<std::string>("type");
            auto sz   = n.second->opts->require_as<size_t>("size");

            ZI_ASSERT(sz>0);

            if ( type == "input" )
            {
                n.second->nodes = std::make_unique<input_nodes>
                    (sz,n.second->fsize,*n.second->opts,tm_);
            }
            else if ( (type == "sum") || (type == "transfer") )
            {
                n.second->nodes
                    = std::make_unique<transfer_nodes>
                    (sz,n.second->fsize,*n.second->opts,tm_,
                     n.second->out.size()==0);
            }
            else
            {
                throw std::logic_error(HERE() + "unknown nodes type: " + type);
            }

            n.second->opts = nullptr;
        }
    }


    void add_edges( options const & op )
    {
        auto name = op.require_as<std::string>("name");
        auto type = op.require_as<std::string>("type");
        auto in   = op.require_as<std::string>("input");
        auto out  = op.require_as<std::string>("output");

        ZI_ASSERT(edges_.count(name)==0);
        ZI_ASSERT(nodes_.count(in)&&nodes_.count(out));

        nedges * es = new nedges;
        es->opts = &op;
        es->in   = nodes_[in];
        es->out  = nodes_[out];
        nodes_[in]->out.push_back(es);
        nodes_[out]->in.push_back(es);

        edges_[name] = es;

        if ( type == "max_filter" )
        {
            es->width  = op.require_as<ovec3i>("size");
            es->stride = op.require_as<ovec3i>("stride");
        }
        else if ( type == "conv" )
        {
            es->width  = op.require_as<ovec3i>("size");
            es->stride = op.optional_as<ovec3i>("stride", "1,1,1");
        }
        else if ( type == "dummy" )
        {
        }
        else
        {
            throw std::logic_error(HERE() + "unknown nodes type: " + type);
        }

    }


public:
    network( std::vector<options> const & ns,
             std::vector<options> const & es,
             vec3i const & outsz,
             size_t n_threads = 1 )
        : tm_(n_threads)
    {
        for ( auto& n: ns ) add_nodes(n);
        for ( auto& e: es ) add_edges(e);
        init(outsz);
        create_nodes();
        create_edges();
    }

    void set_eta( double eta )
    {
        for ( auto & e: edges_ ) e.second->edges->set_eta(eta);
        for ( auto & n: nodes_ ) n.second->nodes->set_eta(eta);
    }

    void set_momentum( double mom )
    {
        for ( auto & e: edges_ ) e.second->edges->set_momentum(mom);
        for ( auto & n: nodes_ ) n.second->nodes->set_momentum(mom);
    }

    void set_weight_decay( double wd )
    {
        for ( auto & e: edges_ ) e.second->edges->set_weight_decay(wd);
        for ( auto & n: nodes_ ) n.second->nodes->set_weight_decay(wd);
    }

    vec3i fov() const
    {
        return input_nodes_.begin()->second->fov;
    }

    std::map<std::string, std::vector<cube_p<double>>>
    forward( std::map<std::string, std::vector<cube_p<double>>> && fin )
    {
        ZI_ASSERT(fin.size()==input_nodes_.size());
        for ( auto & in: fin )
        {
            ZI_ASSERT(input_nodes_.count(in.first));

            auto& in_layer = input_nodes_[in.first]->nodes;

            ZI_ASSERT(in_layer->num_in_nodes()==in.second.size());

            for ( size_t i = 0; i < in.second.size(); ++i )
            {
                in_layer->forward(i, std::move(in.second[i]));
            }
        }

        std::map<std::string, std::vector<cube_p<double>>> ret;
        for ( auto & l: output_nodes_ )
        {
            l.second->nodes->wait();
            std::cout << "Collecting for: " << l.first << "\n";
            ret[l.first] = l.second->nodes->get_featuremaps();
        }

        return ret;
    }

    std::map<std::string, std::vector<cube_p<double>>>
    backward( std::map<std::string, std::vector<cube_p<double>>> && fout )
    {
        ZI_ASSERT(fout.size()==input_nodes_.size());
        for ( auto & out: fout )
        {
            ZI_ASSERT(output_nodes_.count(out.first));

            auto& out_layer = output_nodes_[out.first]->nodes;

            ZI_ASSERT(out_layer->num_out_nodes()==out.second.size());

            for ( size_t i = 0; i < out.second.size(); ++i )
            {
                out_layer->backward(i, std::move(out.second[i]));
            }
        }

        std::map<std::string, std::vector<cube_p<double>>> ret;
        for ( auto & l: input_nodes_ )
        {
            l.second->nodes->wait();
            ret[l.first].resize(0);
        }

        return ret;
    }

    std::pair<std::vector<options>,std::vector<options>> serialize() const
    {
        std::pair<std::vector<options>,std::vector<options>> ret;

        for ( auto & n: nodes_ )
            ret.first.push_back(n.second->nodes->serialize());

        for ( auto & e: edges_ )
            ret.second.push_back(e.second->edges->serialize());

        return ret;
    }

    void zap()
    {
        for ( auto & n: nodes_ )
            n.second->nodes->zap();

        for ( auto & e: edges_ )
            e.second->edges->zap();
    }

};


}}} // namespace znn::v4::trivial_fft_network
