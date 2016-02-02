//
// Copyright (C) 2012-2015  Aleksandar Zlateski <zlateski@mit.edu>
// ---------------------------------------------------------------
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.
//
#pragma once

#include "edges.hpp"
#include "input_nodes.hpp"
#include "transfer_nodes.hpp"
#include "maxout_nodes.hpp"
#include "../../initializator/initializators.hpp"
#include "../helpers.hpp"

#include <map>
#include <zi/time.hpp>

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

        bool pool = false;
        bool crop = false;

        nnodes * in;
        nnodes * out;

        options const * opts;

        std::unique_ptr<edges> dedges;
    };

    struct nnodes
    {
        vec3i fov       = vec3i::zero;
        vec3i stride    = vec3i::zero;
        vec3i fsize     = vec3i::zero;

        options const * opts;

        std::unique_ptr<nodes> dnodes;
        std::vector<nedges *> in, out;

        size_t fwd_priority = 0;
        size_t bwd_priority = 0;
    };

private:
    network( network const & ) = delete;
    network& operator=( network const & ) = delete;

    network( network && ) = delete;
    network& operator=( network && ) = delete;

public:
    ~network()
    {
        zap();
        for ( auto& n: nodes_ ) delete n.second;
        for ( auto& e: edges_ ) delete e.second;
    }

private:
    static std::vector<std::map<std::string, std::vector<cube_p<real>>>>
    copy_samples( std::vector<std::map<std::string, std::vector<cube_p<real>>>>
                 const & s )
    {
        std::vector<std::map<std::string, std::vector<cube_p<real>>>>
            ret(s.size());

        for ( size_t i = 0; i < s.size(); ++i )
        {
            for ( auto & x: s[i] )
            {
                for ( auto & y: x.second )
                {
                    ret[i][x.first].push_back(get_copy(*y));
                }
            }
        }
        return ret;
    }

private:
    std::map<std::string, nedges*> edges_;
    std::map<std::string, nnodes*> nodes_;
    std::map<std::string, nnodes*> input_nodes_;
    std::map<std::string, nnodes*> output_nodes_;

    // [kisuklee]
    // This is only temporary implementation and will be removed.
    std::map<std::string, nedges*> phase_dependent_edges_;

    task_manager tm_;

    phase phase_;

#ifdef ZNN_ANALYSE_TASK_MANAGER
    void dump() { tm_.dump(); }
#endif


    void fov_pass(nnodes* n, vec3i fov, const vec3i& fsize )
    {
        if ( n->fov != vec3i::zero )
        {
            ZI_ASSERT(n->fsize==fsize);
            if ( n->fov == fov )
            {
                return;
            }
            else
            {
                fov[0] = std::max(n->fov[0],fov[0]);
                fov[1] = std::max(n->fov[1],fov[1]);
                fov[2] = std::max(n->fov[2],fov[2]);
            }
        }

        for ( auto& e: n->out )
        {
            e->in_fsize = fsize;
        }
        n->fov = fov;
        n->fsize = fsize;
        for ( auto& e: n->in )
        {
            if ( e->pool )
            {
                vec3i new_fov   = fov * e->width;
                vec3i new_fsize = e->width * fsize;
                fov_pass(e->in, new_fov, new_fsize);
            }
            else if ( e->crop )
            {
                // FoV doesn't change
                fov_pass(e->in, fov, fsize + e->width - vec3i::one);
            }
            else
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
            vec3i real_stride = stride;

            if ( n->opts->optional_as<int>("dense",0) )
            {
                real_stride = vec3i::one;
            }

            n->stride = real_stride;

            for ( auto& e: n->out )
            {
                if ( e->pool && (e->stride!=vec3i::one) )
                {
                    UNIMPLEMENTED();
                }

                e->in_stride = real_stride;
                stride_pass(e->out, real_stride * e->stride );
            }
        }
    }

    size_t fwd_priority_pass(nnodes* n)
    {
        if ( n->fwd_priority > 0 )
        {
            return n->fwd_priority;
        }

        size_t p = 0;

        for ( auto& e: n->in )
        {
            p = std::max(p, fwd_priority_pass(e->in));
        }

        n->fwd_priority = p + 1;
        return n->fwd_priority;
    }

    size_t bwd_priority_pass(nnodes* n)
    {
        if ( n->bwd_priority > 0 )
        {
            return n->bwd_priority;
        }

        size_t p = n->fwd_priority;

        for ( auto& e: n->out )
        {
            p = std::max(p, bwd_priority_pass(e->out));
        }

        n->bwd_priority = p + 1;
        return n->bwd_priority;
    }

    // [kisuklee]
    // This should be modified later to deal with multiple output layers
    // with different size.
    void set_patch_size( vec3i const& outsz )
    {
        real s = outsz[0]*outsz[1]*outsz[2];

        for ( auto& n: nodes_ )
            n.second->dnodes->set_patch_size(s);
        for ( auto& e: edges_ )
            e.second->dedges->set_patch_size(s);
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

        for ( auto& o: output_nodes_ )
            fwd_priority_pass(o.second);
        for ( auto& o: input_nodes_ )
            bwd_priority_pass(o.second);

        // for ( auto& o: nodes_ )
        // {
        //     std::cout << "NODE GROUP: " << o.first << "\n    "
        //               << "FOV: " << o.second->fov << "\n    "
        //               << "STRIDE: " << o.second->stride << "\n    "
        //               << "SIZE: " << o.second->fsize << '\n';
        // }

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
        nnodes* ns = new nnodes;
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
            auto opts = e.second->opts;
            auto type = opts->require_as<std::string>("type");
            nodes * in  = e.second->in->dnodes.get();
            nodes * out = e.second->out->dnodes.get();

            if ( type == "max_filter" )
            {
                e.second->dedges = std::make_unique<edges>
                    ( in, out, *opts, e.second->in_stride,
                      e.second->in_fsize, tm_, edges::max_pooling_tag() );
            }
            else if ( type == "max_pool" )
            {
                e.second->dedges = std::make_unique<edges>
                    ( in, out, *opts,
                      e.second->in_fsize, tm_, edges::real_pooling_tag() );
            }
            else if ( type == "conv" )
            {
                e.second->dedges = std::make_unique<edges>
                    ( in, out, *opts, e.second->in_stride,
                      e.second->in_fsize, tm_, edges::filter_tag() );
            }
            else if ( type == "dropout" )
            {
                // [kisuklee]
                // This version of dropout isn't actually disabling individual
                // nodes, but making a random binary dropout masks for each
                // node. This is the version that was implemented in v1, and
                // the effectiveness is yet to be proven.

                e.second->dedges = std::make_unique<edges>
                    ( in, out, *opts, e.second->in_fsize,
                      tm_, phase_, edges::dropout_tag() );
            }
            else if ( type == "crop" )
            {
                e.second->dedges = std::make_unique<edges>
                    ( in, out, *opts, tm_, edges::crop_tag() );
            }
            else if ( type == "maxout")
            {
                // sanity check
                STRONG_ASSERT(dynamic_cast<maxout_nodes*>(out));

                e.second->dedges = std::make_unique<edges>
                    ( in, out, *opts, tm_, edges::maxout_tag() );
            }
            else if ( type == "dummy" )
            {
                e.second->dedges = std::make_unique<edges>
                    ( in, out, *opts, tm_, edges::dummy_tag() );
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
        std::map<size_t,size_t> fwd_pts;
        std::map<size_t,size_t> bwd_pts;

        for ( auto & n: nodes_ )
        {
            auto type = n.second->opts->require_as<std::string>("type");
            auto sz   = n.second->opts->require_as<size_t>("size");

            size_t fwd_p = n.second->fwd_priority * 1024
                + fwd_pts[n.second->fwd_priority];
            ++fwd_pts[n.second->fwd_priority];

            size_t bwd_p = n.second->bwd_priority * 1024
                + bwd_pts[n.second->bwd_priority];
            ++bwd_pts[n.second->bwd_priority];


            ZI_ASSERT(sz>0);

            if ( type == "input" )
            {
                n.second->dnodes = std::make_unique<input_nodes>
                    (sz,n.second->fsize,*n.second->opts,tm_,fwd_p,bwd_p);
            }
            else if ( (type == "sum") || (type == "transfer") )
            {
                n.second->dnodes = std::make_unique<transfer_nodes>
                    ( sz, n.second->fsize, *n.second->opts, tm_,
                      fwd_p,bwd_p,n.second->out.size()==0 );
            }
            else if ( type == "maxout" )
            {
                n.second->dnodes = std::make_unique<maxout_nodes>
                    ( sz, n.second->fsize, *n.second->opts, tm_,
                      fwd_p,bwd_p,n.second->out.size()==0 );
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
        es->opts    = &op;
        es->in      = nodes_[in];
        es->out     = nodes_[out];
        es->pool    = false;
        es->crop    = false;
        es->stride  = vec3i::one;
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
        else if ( type == "max_pool" )
        {
            es->width  = op.require_as<ovec3i>("size");
            es->pool   = true;
        }
        else if ( type == "dropout" )
        {
            phase_dependent_edges_[name] = es;
        }
        else if ( type == "crop" )
        {
            auto off = op.require_as<ovec3i>("offset");
            es->width   = off + off + vec3i::one;
            es->crop    = true;
        }
        else if ( type == "maxout" )
        {
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
             size_t n_threads = 1,
             phase phs = phase::TRAIN )
        : tm_(n_threads)
        , phase_(phs)
    {
        for ( auto& n: ns ) add_nodes(n);
        for ( auto& e: es ) add_edges(e);
        init(outsz);
        create_nodes();
        create_edges();

        // minibatch averaging
        set_patch_size(outsz);
    }

    void set_eta( real eta )
    {
        zap();
        for ( auto & e: edges_ ) e.second->dedges->set_eta(eta);
        for ( auto & n: nodes_ ) n.second->dnodes->set_eta(eta);
    }

    void set_momentum( real mom )
    {
        zap();
        for ( auto & e: edges_ ) e.second->dedges->set_momentum(mom);
        for ( auto & n: nodes_ ) n.second->dnodes->set_momentum(mom);
    }

    void set_weight_decay( real wd )
    {
        zap();
        for ( auto & e: edges_ ) e.second->dedges->set_weight_decay(wd);
        for ( auto & n: nodes_ ) n.second->dnodes->set_weight_decay(wd);
    }

    vec3i fov() const
    {
        return input_nodes_.begin()->second->fov;
    }

    // [kisuklee]
    // This is only temporary implementation and will be removed.
    void set_phase( phase phs = phase::TRAIN )
    {
        zap();
        for ( auto & e: phase_dependent_edges_ )
            e.second->dedges->set_phase(phs);
    }

    std::map<std::string, std::pair<vec3i,size_t>> inputs() const
    {
        std::map<std::string, std::pair<vec3i,size_t>> ret;
        for ( auto & in: input_nodes_ )
        {
            ret[in.first] = { in.second->fsize,
                              in.second->dnodes->num_in_nodes() };
        }
        return ret;
    }

    std::map<std::string, std::pair<vec3i,size_t>> outputs() const
    {
        std::map<std::string, std::pair<vec3i,size_t>> ret;
        for ( auto & in: output_nodes_ )
        {
            ret[in.first] = { in.second->fsize,
                              in.second->dnodes->num_out_nodes() };
        }
        return ret;
    }

    std::map<std::string, std::vector<cube_p<real>>>
    forward( std::map<std::string, std::vector<cube_p<real>>> && fin )
    {
        ZI_ASSERT(fin.size()==input_nodes_.size());
        for ( auto & in: fin )
        {
            ZI_ASSERT(input_nodes_.count(in.first));

            auto& in_layer = input_nodes_[in.first]->dnodes;

            ZI_ASSERT(in_layer->num_in_nodes()==in.second.size());

            for ( size_t i = 0; i < in.second.size(); ++i )
            {
                in_layer->forward(i, std::move(in.second[i]));
            }
        }

        std::map<std::string, std::vector<cube_p<real>>> ret;
        for ( auto & l: output_nodes_ )
        {
            l.second->dnodes->wait();
            ret[l.first] = l.second->dnodes->get_featuremaps();
        }

        return ret;
    }

    std::map<std::string, std::vector<cube_p<real>>>
    backward( std::map<std::string, std::vector<cube_p<real>>> && fout )
    {
        ZI_ASSERT(fout.size()==input_nodes_.size());
        for ( auto & out: fout )
        {
            ZI_ASSERT(output_nodes_.count(out.first));

            auto& out_layer = output_nodes_[out.first]->dnodes;

            ZI_ASSERT(out_layer->num_out_nodes()==out.second.size());

            for ( size_t i = 0; i < out.second.size(); ++i )
            {
                out_layer->backward(i, std::move(out.second[i]));
            }
        }

        std::map<std::string, std::vector<cube_p<real>>> ret;
        for ( auto & l: input_nodes_ )
        {
            l.second->dnodes->wait();
            ret[l.first].resize(0);
        }

        return ret;
    }

    std::pair<std::vector<options>,std::vector<options>> serialize()
    {
        zap();
        std::pair<std::vector<options>,std::vector<options>> ret;

        for ( auto & n: nodes_ )
            ret.first.push_back(n.second->dnodes->serialize());

        for ( auto & e: edges_ )
            ret.second.push_back(e.second->dedges->serialize());

        return ret;
    }

    void zap()
    {
        for ( auto & n: nodes_ )
            n.second->dnodes->zap();

        for ( auto & e: edges_ )
            e.second->dedges->zap();
    }

    static void optimize( std::vector<options> & ns,
                          std::vector<options> & es,
                          vec3i const & outsz,
                          size_t n_threads = 1,
                          size_t rounds = 10)
    {
        std::vector<options*> edge_groups;
        for ( auto & e: es )
        {
            if ( e.require_as<std::string>("type") == "conv" )
            {
                edge_groups.push_back(&e);
                e.push("fft",1);
            }
        }

        std::cout << "Total of " << edge_groups.size()
                  << " to optimize\n\n";

        // generate 10 inputs and outputs
        network net(ns,es,outsz,n_threads);

        std::vector<std::map<std::string, std::vector<cube_p<real>>>>
            allins, allouts;


        std::cout << "Create samples...";

        std::tie(allins, allouts) = generate_inout(rounds,net);

        std::cout << "DONE\nFFT Warmup..." << std::flush;

        {
            auto is = copy_samples(allins);
            auto os = copy_samples(allouts);
            net.forward(std::move(is[0]));
            net.backward(std::move(os[0]));
            net.zap();
        }

        std::cout << "DONE\nTrying all FFTs..." << std::flush;

        real tot_time = 0;

        {
            network net(ns,es,outsz,n_threads);

            auto is = copy_samples(allins);
            auto os = copy_samples(allouts);

            zi::wall_timer wt;
            net.forward(std::move(is[0]));

            wt.reset();

            for ( size_t i = 0; i < rounds-1; ++i )
            {
                net.backward(std::move(os[i]));
                net.forward(std::move(is[i+1]));
            }

            tot_time = wt.elapsed<real>();

            //net.backward(std::move(os[rounds-1]));
            net.zap();

#ifdef ZNN_ANALYSE_TASK_MANAGER
            std::cout << "HERE";
            net.dump();
#endif


            std::cout << (tot_time/(rounds-1)) << " secs" << std::endl;
        }

        //std::vector<options*> edge_groups;
        for ( auto & e: edge_groups )
        {
            std::cout << "Trying edge group: "
                      << e->require_as<std::string>("name")
                      << " ..." << std::flush;
            e->push("fft","0");

            network net(ns,es,outsz,n_threads);

            auto is = copy_samples(allins);
            auto os = copy_samples(allouts);

            zi::wall_timer wt;
            net.forward(std::move(is[0]));

            wt.reset();

            for ( size_t i = 0; i < rounds-1; ++i )
            {
                net.backward(std::move(os[i]));
                net.forward(std::move(is[i+1]));
            }

            //net.backward(std::move(os[rounds-1]));

            real my_time = wt.elapsed<real>();

            std::cout <<  (my_time/(rounds-1)) << " secs" << std::endl;

            if ( my_time < tot_time )
            {
                tot_time = my_time;
                std::cout << "   will use direct convolution" << std::endl;
            }
            else
            {
                std::cout << "   will use FFT convolution" << std::endl;
                e->push("fft","1");
            }
        }
    }

    static void optimize_forward( std::vector<options> & ns,
                                  std::vector<options> & es,
                                  vec3i const & outsz,
                                  size_t n_threads = 1,
                                  size_t rounds = 10 )
    {
        std::vector<options*> edge_groups;
        for ( auto & e: es )
        {
            if ( e.require_as<std::string>("type") == "conv" )
            {
                edge_groups.push_back(&e);
                e.push("fft",1);
            }
        }

        std::cout << "Total of " << edge_groups.size()
                  << " to optimize\n\n";

        // generate 10 inputs and outputs
        network net(ns,es,outsz,n_threads);
        auto ins  = net.inputs();
        auto outs = net.outputs();

        std::vector<std::map<std::string, std::vector<cube_p<real>>>>
            allins, allouts;

        std::cout << "Create samples...";

        std::tie(allins, allouts) = generate_inout(rounds,net);

        std::cout << "DONE\nFFT Warmup..." << std::flush;

        {
            auto is = copy_samples(allins);
            auto os = copy_samples(allouts);
            net.forward(std::move(is[0]));
            //net.backward(std::move(os[0]));
            net.zap();
        }

        std::cout << "DONE\nTrying all FFTs..." << std::flush;

        real tot_time = 0;

        {
            network net(ns,es,outsz,n_threads);

            auto is = copy_samples(allins);
            auto os = copy_samples(allouts);

            zi::wall_timer wt;
            net.forward(std::move(is[0]));

            wt.reset();

            for ( size_t i = 0; i < rounds-1; ++i )
            {
                //net.backward(std::move(os[i]));
                net.forward(std::move(is[i+1]));
            }

            //net.backward(std::move(os[rounds-1]));

            tot_time = wt.elapsed<real>();

            net.zap();

            std::cout << (tot_time/(rounds-1)) << " secs" << std::endl;
        }

        //std::vector<options*> edge_groups;
        for ( auto & e: edge_groups )
        {
            std::cout << "Trying edge group: "
                      << e->require_as<std::string>("name")
                      << " ..." << std::flush;
            e->push("fft","0");

            network net(ns,es,outsz,n_threads);

            auto is = copy_samples(allins);
            auto os = copy_samples(allouts);

            zi::wall_timer wt;
            net.forward(std::move(is[0]));

            wt.reset();

            for ( size_t i = 0; i < rounds-1; ++i )
            {
                //net.backward(std::move(os[i]));
                net.forward(std::move(is[i+1]));
            }

            //net.backward(std::move(os[rounds-1]));

            real my_time = wt.elapsed<real>();

            std::cout << (my_time/(rounds-1)) << " secs" << std::endl;

            if ( my_time < tot_time )
            {
                tot_time = my_time;
                std::cout << "   will use direct convolution" << std::endl;
            }
            else
            {
                std::cout << "   will use FFT convolution" << std::endl;
                e->push("fft","1");
            }
        }
    }

    static void force_fft( std::vector<options> & es )
    {
        for ( auto & e: es )
        {
            if ( e.require_as<std::string>("type") == "conv" )
            {
                e.push("fft",1);
            }
        }
    }

    static std::pair<double,double> speed_test( std::vector<options> & ns,
                                                std::vector<options> & es,
                                                vec3i const & outsz,
                                                size_t n_threads = 1,
                                                size_t rounds = 10,
                                                size_t warmup = 1 )
    {
        std::vector<options*> edge_groups;
        for ( auto & e: es )
        {
            if ( e.require_as<std::string>("type") == "conv" )
            {
                edge_groups.push_back(&e);
                e.push("fft",1);
            }
        }

        std::vector<std::map<std::string, std::vector<cube_p<real>>>>
            allins, allouts;

        {
            network net(ns,es,outsz,n_threads);

            std::cout << "Create samples...";

            std::tie(allins, allouts) =
                generate_inout(std::max(rounds+1,warmup+1),net);

            std::cout << "DONE\n  Warmup......" << std::flush;

            auto is = copy_samples(allins);
            auto os = copy_samples(allouts);
            net.forward(std::move(is[0]));

            for ( size_t i = 0; i < warmup; ++i )
            {
                net.backward(std::move(os[i]));
                net.forward(std::move(is[i+1]));
            }

            net.zap();
        }

        std::cout << "DONE\n  Measuring..." << std::flush;

        {
#ifdef ZNN_USE_MKL_DIRECT_CONV
            conv_plans.unlock();
#endif
            network net(ns,es,outsz,n_threads);

            auto is = copy_samples(allins);
            auto os = copy_samples(allouts);

            std::vector<double> times(rounds);
            zi::wall_timer wt;

            net.forward(std::move(is[0]));

            wt.reset();

            for ( size_t i = 0; i < rounds; ++i )
            {
                net.backward(std::move(os[i]));
                net.forward(std::move(is[i+1]));
                times[i] = wt.lap<double>();
            }

            auto ret = measured(times);
            std::cout << ret.first << " +/- " << ret.second << " secs" << std::endl;
            net.zap();

#ifdef ZNN_USE_MKL_DIRECT_CONV
            conv_plans.lock();
#endif
            return ret;
        }
    }


};


}}} // namespace znn::v4::parallel_network
