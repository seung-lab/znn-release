#pragma once

#include "../../assert.hpp"
#include "../../cube/cube.hpp"
#include "../../cube/cube_operators.hpp"
#include "../../pooling/pooling.hpp"
#include "../../initializator/initializators.hpp"
#include "../../options/options.hpp"
#include "../../convolution/convolution.hpp"
#include "../../transfer_function/transfer_functions.hpp"
#include "../../utils/dispatcher.hpp"
#include "../../utils/accumulator.hpp"
#include "../../utils/waiter.hpp"
#include "../../utils/task_manager.hpp"
#include "../../types.hpp"
#include "../bias.hpp"
#include "../filter.hpp"
#include "utils.hpp"

#include <map>
#include <string>
#include <vector>

namespace znn { namespace v4 { namespace trivial_forward_network {

// Forward definition
class edge;

class nodes
{
private:
    vec3i const fsize_;

protected:
    nodes( vec3i const & fsize ): fsize_(fsize) {}

public:
    vec3i const & fsize() const { return fsize_; }

public:
    virtual ~nodes() {}

    // receive a featuremap for the i-th input
    // featuremap is absorbed
    virtual void forward(size_t, cube_p<real>&&)
    { UNIMPLEMENTED(); }

    // receive a featuremap for the i-th input
    // featuremap is absorbed
    virtual void forward(size_t, size_t, cube_p<complex>&&)
    { UNIMPLEMENTED(); }

    virtual std::vector<cube_p<real>>& get_featuremaps()
    { UNIMPLEMENTED(); }

    virtual size_t num_out_nodes()
    { UNIMPLEMENTED(); }

    virtual size_t num_in_nodes()
    { UNIMPLEMENTED(); }

    virtual void attach_out_edge(size_t, edge*)
    { UNIMPLEMENTED(); }

    virtual void attach_in_edge(size_t, edge*)
    { UNIMPLEMENTED(); }

    virtual size_t attach_out_fft_edge(size_t, edge*)
    { UNIMPLEMENTED(); }

    virtual size_t attach_in_fft_edge(size_t, edge*, vec3i const &)
    { UNIMPLEMENTED(); }

};



class edge
{
public:
    virtual ~edge() {}

    // perform forward computation
    // can't modify the featuremap
    virtual void forward( ccube_p<real> const & )
    { UNIMPLEMENTED(); }

    virtual void forward( ccube_p<complex> const & )
    { UNIMPLEMENTED(); }

};

class edge_base: public virtual edge
{
protected:
    nodes * in_nodes ;
    size_t  in_num   ;
    nodes * out_nodes;
    size_t  out_num  ;

public:
    edge_base( nodes* in,
               size_t inn,
               nodes* out,
               size_t outn )
        : in_nodes(in)
        , in_num(inn)
        , out_nodes(out)
        , out_num(outn)
    {
    }
};



template< typename E >
class edge_of: public edge_base
{
private:
    E impl;

public:
    template<class... Args>
    edge_of( nodes* in,
             size_t inn,
             nodes* out,
             size_t outn,
             Args&&... args )
        : edge_base(in, inn, out, outn)
        , impl(std::forward<Args>(args)...)
    {
        // attach myself
        in->attach_out_edge(inn, this);
        out->attach_in_edge(outn, this);
    }

    void forward( ccube_p<real> const & f ) override
    {
        out_nodes->forward(out_num, impl.forward(f));
    }

};

struct dummy_edge
{
    cube_p<real> forward( ccube_p<real> const & f )
    {
        return get_copy(*f);
    }
};

class max_pooling_edge
{
private:
    vec3i filter_size;
    vec3i filter_stride;

    cube_p<int> indices;
    vec3i       insize ;

public:
    max_pooling_edge( vec3i const & size,
                      vec3i const & stride )
        : filter_size(size), filter_stride(stride)
    {
    }

    cube_p<real> forward( ccube_p<real> const & f )
    {
        insize = size(*f);
        auto r = pooling_filter(get_copy(*f),
                                [](real a, real b){ return a>b; },
                                filter_size,
                                filter_stride);
        indices = r.second;
        return r.first;
    }

};


class filter_edge
{
private:
    vec3i    filter_stride;
    filter & filter_;

    ccube_p<real> last_input;

public:
    filter_edge( vec3i const & stride, filter & f )
        : filter_stride(stride), filter_(f)
    {
    }

    cube_p<real> forward( ccube_p<real> const & f )
    {
        last_input = f;
        return convolve_sparse(*f, filter_.W(), filter_stride);
    }

};

class fft_filter_edge: public edge_base
{
private:
    vec3i    filter_stride;
    filter & filter_;

    ccube_p<complex> w_fft;
    ccube_p<complex> last_input;

    size_t fwd_bucket_;
    size_t bwd_bucket_;

public:
    fft_filter_edge( nodes* in,
                     size_t inn,
                     nodes* out,
                     size_t outn,
                     vec3i const & stride,
                     filter & flt )
        : edge_base(in, inn, out, outn)
        , filter_stride(stride)
        , filter_(flt)
    {
        // attach myself

        bwd_bucket_ = in->attach_out_fft_edge(inn, this);
        fwd_bucket_ = out->attach_in_fft_edge(outn, this, in->fsize());
    }

    void forward( ccube_p<complex> const & f ) override
    {
        last_input = f;
        // TODO(zlateski): WTH was happening with sparse_exploce before
        //                 when I had to use sparse_explode_slow,
        //                 ony happened on my laptop
        auto w_tmp = sparse_explode_slow(filter_.W(), filter_stride,
                                         in_nodes->fsize());
        w_fft = fftw::forward(std::move(w_tmp));
        auto fw = *w_fft * *f;
        out_nodes->forward(out_num, fwd_bucket_, std::move(fw));
    }

};


class edges
{
public:
    virtual ~edges() {}
};

class dummy_edges: public edges
{
private:
    options                                    options_;
    std::vector<std::unique_ptr<edge>> edges_  ;

public:
    dummy_edges( nodes * in,
                 nodes * out,
                 options const & opts )
        : options_(opts)
    {
        ZI_ASSERT(in->num_out_nodes()==out->num_in_nodes());

        size_t n = in->num_out_nodes();
        edges_.resize(n);

        for ( size_t i = 0; i < n; ++i )
        {
            edges_[i] = std::make_unique<edge_of<dummy_edge>>
                (in, i, out, i);
        }
    }

};

class max_pooling_edges: public edges
{
private:
    options                                    options_;
    std::vector<std::unique_ptr<edge>> edges_  ;

public:
    max_pooling_edges( nodes * in,
                       nodes * out,
                       options const & opts,
                       vec3i const & stride )
        : options_(opts)
    {
        ZI_ASSERT(in->num_out_nodes()==out->num_in_nodes());

        size_t n = in->num_out_nodes();
        edges_.resize(n);

        auto sz     = opts.require_as<ovec3i>("size");

        for ( size_t i = 0; i < n; ++i )
        {
            edges_[i]
                = std::make_unique<edge_of<max_pooling_edge>>
                (in, i, out, i, sz, stride);
        }
    }

};

class filter_edges: public edges
{
private:
    options                                           options_;
    std::vector<std::unique_ptr<filter>>              filters_;
    std::vector<std::unique_ptr<edge>>                edges_  ;
    vec3i                                             size_   ;

public:
    filter_edges( nodes * in,
                  nodes * out,
                  options const & opts,
                  vec3i const & stride )
        : options_(opts)
    {
        size_t n = in->num_out_nodes();
        size_t m = out->num_in_nodes();

        ZI_ASSERT((n>0)&&m>0);

        edges_.resize(n*m);
        filters_.resize(n*m);

        real eta    = opts.optional_as<real>("eta", 0.1);
        real mom    = opts.optional_as<real>("momentum", 0.0);
        real wd     = opts.optional_as<real>("weight_decay", 0.0);
        auto   sz     = opts.require_as<ovec3i>("size");

        size_ = sz;

        for ( size_t i = 0, k = 0; i < n; ++i )
        {
            for ( size_t j = 0; j < m; ++j, ++k )
            {
                filters_[k] = std::make_unique<filter>(sz, eta, mom, wd);
                edges_[k]
                    = std::make_unique<edge_of<filter_edge>>
                    (in, i, out, j, stride, *filters_[k]);
            }
        }

        std::string filter_values;

        opts.dump();

        if ( opts.contains("filters") )
        {
            filter_values = opts.require_as<std::string>("filters");
        }
        else
        {
            size_t n_values = n*m*size_[0]*size_[1]*size_[2];
            real * filters_raw = new real[n_values];

            auto initf = get_initializator(opts);


            initf->initialize( filters_raw, n*m*size_[0]*size_[1]*size_[2] );
            delete [] filters_raw;

            filter_values = std::string( reinterpret_cast<char*>(filters_raw),
                                         sizeof(real) * n_values );
        }

        load_filters(filters_, size_, filter_values);
    }

};


class fft_filter_edges: public edges
{
private:
    options                                           options_;
    std::vector<std::unique_ptr<filter>>              filters_;
    std::vector<std::unique_ptr<edge>>                edges_  ;
    vec3i                                             size_   ;

public:
    fft_filter_edges( nodes * in,
                      nodes * out,
                      options const & opts,
                      vec3i const & stride )
        : options_(opts)
    {
        size_t n = in->num_out_nodes();
        size_t m = out->num_in_nodes();

        ZI_ASSERT((n>0)&&m>0);

        edges_.resize(n*m);
        filters_.resize(n*m);

        real eta    = opts.optional_as<real>("eta", 0.1);
        real mom    = opts.optional_as<real>("momentum", 0.0);
        real wd     = opts.optional_as<real>("weight_decay", 0.0);
        auto   sz     = opts.require_as<ovec3i>("size");

        size_ = sz;

        for ( size_t i = 0, k = 0; i < n; ++i )
        {
            for ( size_t j = 0; j < m; ++j, ++k )
            {
                filters_[k] = std::make_unique<filter>(sz, eta, mom, wd);
                edges_[k]
                    = std::make_unique<fft_filter_edge>
                    (in, i, out, j, stride, *filters_[k]);
            }
        }

        std::string filter_values;

        opts.dump();

        if ( opts.contains("filters") )
        {
            filter_values = opts.require_as<std::string>("filters");
        }
        else
        {
            size_t n_values = n*m*size_[0]*size_[1]*size_[2];
            real * filters_raw = new real[n_values];

            auto initf = get_initializator(opts);

            initf->initialize( filters_raw, n*m*size_[0]*size_[1]*size_[2] );
            delete [] filters_raw;

            filter_values = std::string( reinterpret_cast<char*>(filters_raw),
                                         sizeof(real) * n_values );
        }

        load_filters(filters_, size_, filter_values);
    }


};




class input_nodes: public nodes
{
private:
    size_t                                          size_   ;
    dispatcher_group<forward_dispatcher<edge,edge>> outputs_;
    options                                         opts_   ;

public:
    input_nodes(size_t s, vec3i const & fsize, options const & op)
        : nodes(fsize)
        , size_(s)
        , outputs_(s)
        , opts_(op)
    {}

    void forward(size_t n, cube_p<real>&& f) override
    {
        ZI_ASSERT(n<size_);
        outputs_.dispatch(n,f);
    }

    size_t num_out_nodes() override { return size_; }
    size_t num_in_nodes()  override { return size_; }

    void attach_out_edge(size_t i, edge* e) override
    {
        ZI_ASSERT(i<size_);
        outputs_.sign_up(i,e);
    }

    size_t attach_out_fft_edge(size_t i, edge* e) override
    {
        ZI_ASSERT(i<size_);
        outputs_.sign_up(i,nodes::fsize(),e);
        return 0;
    }

};


class summing_nodes: public nodes
{
private:
    size_t                                  size_    ;

    dispatcher_group<forward_dispatcher<edge,edge>>    fwd_dispatch_;
    dispatcher_group<backward_dispatcher<edge,edge>>   bwd_dispatch_;
    std::vector<std::unique_ptr<forward_accumulator>>  fwd_accumulators_;
    std::vector<std::unique_ptr<backward_accumulator>> bwd_accumulators_;

    std::vector<cube_p<real>>             fs_      ;
    std::vector<cube_p<real>>             gs_      ;
    options                                 opts_    ;

public:
    summing_nodes(size_t s, vec3i const & fsize, options const & op)
        : nodes(fsize)
        , size_(s)
        , fwd_dispatch_(s)
        , bwd_dispatch_(s)
        , fwd_accumulators_(s)
        , bwd_accumulators_(s)
        , fs_(s)
        , gs_(s)
        , opts_(op)
    {
        for ( size_t i = 0; i < s; ++i )
        {
            fwd_accumulators_[i]
                = std::make_unique<forward_accumulator>(fsize);
            bwd_accumulators_[i]
                = std::make_unique<backward_accumulator>(fsize);
        }
    }

    std::vector<cube_p<real>>& get_featuremaps() override
    {
        return fs_;
    }

    void forward(size_t n, cube_p<real>&& f) override
    {
        ZI_ASSERT(n<size_);
        if ( fwd_accumulators_[n]->add(std::move(f)) )
        {
            fs_[n] = fwd_accumulators_[n]->reset();
            fwd_dispatch_.dispatch(n, fs_[n]);
            fs_[n].reset();
        }
    }

    void forward(size_t n, size_t b, cube_p<complex>&& f) override
    {
        ZI_ASSERT(n<size_);
        if ( fwd_accumulators_[n]->add(b, std::move(f)) )
        {
            fs_[n] = fwd_accumulators_[n]->reset();
            fwd_dispatch_.dispatch(n, fs_[n]);
            fs_[n].reset();
        }
    }

    size_t num_out_nodes() override { return size_; }
    size_t num_in_nodes()  override { return size_; }

    void attach_in_edge(size_t i, edge* e) override
    {
        ZI_ASSERT(i<size_);
        bwd_dispatch_.sign_up(i,e);
        fwd_accumulators_[i]->grow(1);
    }

    void attach_out_edge(size_t i, edge* e) override
    {
        ZI_ASSERT(i<size_);
        fwd_dispatch_.sign_up(i,e);
        bwd_accumulators_[i]->grow(1);
    }

    size_t attach_out_fft_edge(size_t n, edge* e) override
    {
        ZI_ASSERT(n<size_);
        fwd_dispatch_.sign_up(n,nodes::fsize(),e);
        return bwd_accumulators_[n]->grow_fft(nodes::fsize(),1);
    }

    size_t attach_in_fft_edge(size_t n, edge* e, vec3i const & s) override
    {
        ZI_ASSERT(n<size_);
        bwd_dispatch_.sign_up(n,s,e);
        return fwd_accumulators_[n]->grow_fft(s,1);
    }

};


class transfer_nodes: public nodes
{
private:
    size_t                                  size_    ;
    std::vector<std::unique_ptr<bias>>      biases_  ;
    transfer_function                       func_    ;

    dispatcher_group<forward_dispatcher<edge,edge>>  fwd_dispatch_;
    dispatcher_group<backward_dispatcher<edge,edge>> bwd_dispatch_;

    std::vector<std::unique_ptr<forward_accumulator>>  fwd_accumulators_;
    std::vector<std::unique_ptr<backward_accumulator>> bwd_accumulators_;

    std::vector<cube_p<real>>             fs_      ;
    std::vector<cube_p<real>>             gs_      ;
    options                                 options_ ;

public:
    transfer_nodes( vec3i const & fsize, options const & opts )
        : nodes(fsize)
        , size_(opts.require_as<size_t>("size"))
        , biases_(size_)
        , func_()
        , fwd_dispatch_(size_)
        , bwd_dispatch_(size_)
        , fwd_accumulators_(size_)
        , bwd_accumulators_(size_)
        , fs_(size_)
        , gs_(size_)
        , options_(opts)
    {
        for ( size_t i = 0; i < size_; ++i )
        {
            fwd_accumulators_[i]
                = std::make_unique<forward_accumulator>(fsize);
            bwd_accumulators_[i]
                = std::make_unique<backward_accumulator>(fsize);
        }

        func_ = get_transfer_function(opts);

        // initialize biases

        real eta = opts.optional_as<real>("eta", 0.1);
        real mom = opts.optional_as<real>("momentum", 0.0);
        real wd  = opts.optional_as<real>("weight_decay", 0.0);

        for ( auto& b: biases_ )
        {
            b = std::make_unique<bias>(eta, mom, wd);
        }

        std::string bias_values;

        if ( opts.contains("biases") )
        {
            bias_values = opts.require_as<std::string>("biases");
        }
        else
        {
            real biases_raw[size_];
            if ( opts.contains("init") )
            {
                auto initf = get_initializator(opts);
                initf->initialize( biases_raw, size_ );
            }
            else
            {
                std::fill_n(biases_raw, size_, 0);
            }

            bias_values = std::string( reinterpret_cast<char*>(biases_raw),
                                       sizeof(real) * size_ );
        }

        load_biases(biases_, bias_values);
    }


    std::vector<cube_p<real>>& get_featuremaps() override
    {
        return fs_;
    }

    void forward(size_t n, cube_p<real>&& f) override
    {
        ZI_ASSERT(n<size_);
        if ( fwd_accumulators_[n]->add(std::move(f)) )
        {
            fs_[n] = fwd_accumulators_[n]->reset();
            func_.apply(*fs_[n], biases_[n]->b());
            fwd_dispatch_.dispatch(n,fs_[n]);
        }
    }

    void forward(size_t n, size_t b, cube_p<complex>&& f) override
    {
        ZI_ASSERT(n<size_);
        if ( fwd_accumulators_[n]->add(b, std::move(f)) )
        {
            fs_[n] = fwd_accumulators_[n]->reset();
            func_.apply(*fs_[n], biases_[n]->b());
            fwd_dispatch_.dispatch(n, fs_[n]);
        }
    }


    size_t num_out_nodes() override { return size_; }
    size_t num_in_nodes()  override { return size_; }

    void attach_in_edge(size_t i, edge* e) override
    {
        ZI_ASSERT(i<size_);
        bwd_dispatch_.sign_up(i,e);
        fwd_accumulators_[i]->grow(1);
    }

    void attach_out_edge(size_t i, edge* e) override
    {
        ZI_ASSERT(i<size_);
        fwd_dispatch_.sign_up(i,e);
        bwd_accumulators_[i]->grow(1);
    }


    size_t attach_out_fft_edge(size_t n, edge* e) override
    {
        ZI_ASSERT(n<size_);
        fwd_dispatch_.sign_up(n,nodes::fsize(),e);
        return bwd_accumulators_[n]->grow_fft(nodes::fsize(),1);
    }

    size_t attach_in_fft_edge(size_t n, edge* e, vec3i const & s) override
    {
        ZI_ASSERT(n<size_);
        bwd_dispatch_.sign_up(n,s,e);
        return fwd_accumulators_[n]->grow_fft(s,1);
    }

};

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
                if ( e->pool )
                {
                    vec3i new_fov   = fov * e->width;
                    vec3i new_fsize = e->width * fsize;
                    fov_pass(e->in, new_fov, new_fsize);
                }
                else
                {
                    vec3i new_fov   = (fov - vec3i::one) * e->stride + e->width;
                    vec3i new_fsize = (e->width-vec3i::one) * e->in_stride + fsize;
                    fov_pass(e->in, new_fov, new_fsize);
                }
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
                if ( e->pool && (e->stride!=vec3i::one) )
                {
                    std::cout << e->in_stride
                              << ' ' << e->width
                              << ' ' << e->stride << ' ';
                    UNIMPLEMENTED();
                }

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
            nodes * in  = e.second->in->dnodes.get();
            nodes * out = e.second->out->dnodes.get();

            if ( type == "max_filter" )
            {
                e.second->dedges = std::make_unique<max_pooling_edges>
                    ( in, out, *e.second->opts, e.second->in_stride );
            }
            else if ( type == "conv" )
            {
                //e.second->dedges = std::make_unique<filter_edges>
                //    ( in, out, *e.second->opts, e.second->in_stride );
                e.second->dedges = std::make_unique<fft_filter_edges>
                    ( in, out, *e.second->opts, e.second->in_stride );

            }
            else if ( type == "dummy" )
            {
                e.second->dedges = std::make_unique<dummy_edges>
                    ( in, out, *e.second->opts );
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
                n.second->dnodes = std::make_unique<input_nodes>
                    (sz,n.second->fsize,*n.second->opts);
            }
            else if ( type == "sum" )
            {
                n.second->dnodes
                    = std::make_unique<summing_nodes>
                    (sz,n.second->fsize,*n.second->opts);
            }
            else if ( type == "transfer" )
            {
                n.second->dnodes
                    = std::make_unique<transfer_nodes>
                    (n.second->fsize, *n.second->opts);
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
        es->pool   = false;
        es->stride = vec3i::one;

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
             vec3i const & outsz )
    {
        for ( auto& n: ns ) add_nodes(n);
        for ( auto& e: es ) add_edges(e);
        init(outsz);
        create_nodes();
        create_edges();
    }

    vec3i fov() const
    {
        return input_nodes_.begin()->second->fov;
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
            ret[l.first] = l.second->dnodes->get_featuremaps();
        }

        return ret;
    }

};


class layer
{
public:
    size_t n_in ;
    size_t n_out;

    vec3i filter_size;
    vec3i pool_size  ;

    vec3i in_sparse  = vec3i::zero;
    vec3i out_sparse = vec3i::zero;

    vec3i in_size  = vec3i::zero;
    vec3i my_size  = vec3i::zero;
    vec3i out_size = vec3i::zero;

    bool fft = true;

    std::vector<std::vector<cube_p<real>>> ws;
    std::vector<std::vector<cube_p<complex>>> wsf;
    std::vector<real> bs;

    transfer_function tf;

public:
    layer( size_t nin, size_t nout,
           const vec3i& filter_s,
           const vec3i& pool_s )
        : n_in(nin), n_out(nout), filter_size(filter_s),
          pool_size(pool_s), ws(nin), wsf(nin), bs(nout)
    {
        auto initf = std::make_shared<gaussian_init>(0,0.01);
        for ( size_t i = 0; i < nin; ++i )
        {
            ws[i].resize(n_out);
            wsf[i].resize(n_out);
            for ( size_t j = 0; j < nout; ++j )
            {
                ws[i][j] = get_cube<real>(filter_size);
                initf->initialize(ws[i][j]);
            }
        }

        for ( size_t j = 0; j < nout; ++j )
        {
            initf->initialize(&bs[j], 1);
        }

        tf = functions::logistics();
    }

    void warmup()
    {
        auto x = fftw::forward(get_cube<real>(in_size));
        std::cout << size(*x) << std::endl;
        auto pp = fftw::backward(std::move(x), in_size);
        std::cout << size(*pp) << std::endl;
    }

    void prepare_fwd(task_manager& tm)
    {
        waiter wt0; wt0.set(n_in*n_out);
        for ( size_t j = 0; j < n_out; ++j )
        {
            tm.schedule(1, [&,j]() mutable {
                    for ( size_t i = 0; i < n_in; ++i )
                    {
                        auto w_tmp = sparse_explode_slow(*ws[i][j], in_sparse, in_size);
                        wsf[i][j] = fftw::forward(std::move(w_tmp));
                        wt0.one_done();
                    }
                });
        }
        wt0.wait();
    }


    void process_forward( std::vector<std::vector<cube_p<real>>>& in,
                          std::vector<std::vector<cube_p<real>>>& out,
                          task_manager& tm)
    {

        std::vector<std::vector<cube_p<complex>>> fout;

        std::cout << "PROCESS_FORWARD CALLED: " << in.size() << std::endl;

        out.clear();
        out.resize(n_out);
        fout.resize(n_out);
        for ( size_t i = 0; i < n_out; ++i )
        {
            fout[i].resize(in[0].size());
            out[i].resize(in[0].size());
        }


        if ( fft )
        {

            size_t num_fs = in[0].size();

            std::vector<std::vector<cube_p<complex>>> fin;

            fin.resize(in.size());

            for ( size_t i = 0; i < in.size(); ++i )
            {
                fin[i].resize(num_fs);
            }

            waiter wt0; wt0.set(num_fs*in.size());

            for ( size_t i = 0; i < in.size(); ++i )
            {
                for ( size_t j = 0; j < num_fs; ++j )
                {
                    tm.schedule(1, [&in, &fin, &wt0, i, j]() {
                            fin[i][j] = fftw::forward(std::move(in[i][j]));
                            in[i][j].reset();
                            wt0.one_done();
                        });
                }
            }

            wt0.wait();

            waiter wt; wt.set(n_out);

            for ( size_t j = 0; j < n_out; ++j )
            {
                tm.schedule(1, [&,j]() mutable {
                        for ( size_t i = 0; i < n_in; ++i )
                        {
                            auto w_tmp = sparse_explode_slow(*ws[i][j], in_sparse, in_size);
                            auto w_fft = fftw::forward(std::move(w_tmp));
                            w_tmp.reset();

                            //auto w_fft = wsf[i][j]; //fftw::forward(std::move(w_tmp));

                            for ( size_t k = 0; k < num_fs; ++k )
                            {
                                if ( fout[j][k] )
                                {
                                    mad_to(*fin[i][k], *w_fft, *fout[j][k]);
                                }
                                else
                                {
                                    fout[j][k] = *fin[i][k] * *w_fft;
                                }
                            }
                        }

                        std::cout << " output: " << j << " done" << std::endl;
                        wt.one_done();
                    });
            }

            wt.wait();
            std::cout << "HERE\n";

            fin.clear();

            waiter wt2; wt2.set(num_fs*n_out);

            for ( size_t k = 0; k < num_fs; ++k )
            {
                for ( size_t i = 0; i < n_out; ++i )
                {
                    auto fn = [&,k,i]() mutable {
                        auto pp = fftw::backward(std::move(fout[i][k]), in_size);
                        pp = crop_right(*pp, my_size);
                        tf.apply(*pp, bs[i]);
                        out[i][k] = pooling_filter_no_indices
                        ( std::move(pp),
                          [](real a, real b){ return a>b; },
                          pool_size,
                          in_sparse );
                        fout[i][k].reset();
                        wt2.one_done();
                    };
                    tm.schedule(1,fn);
                }
            }

            wt2.wait();

        }
        else
        {
        }
    }

};

class forward_network
{
private:
    std::vector<layer*> layers_;
    task_manager tm;

public:
    forward_network(size_t t): tm(t) {}

    void add_layer( const vec3i& fs, const vec3i& ps, size_t nout )
    {
        size_t nin = 1;
        if ( layers_.size() ) nin = (*layers_.rbegin())->n_out;
        layers_.push_back(new layer(nin, nout, fs, ps));
    }

    vec3i init( const vec3i& out )
    {
        // stride pass;
        vec3i stride(1,1,1);

        for ( auto& l: layers_ )
        {
            l->in_sparse = stride;
            stride *= l->pool_size;
            l->out_sparse = stride;
        }

        vec3i os = out;

        for ( size_t i = layers_.size(); i > 0; --i )
        {
            layer* l = layers_[i-1];
            l->out_size = os;
            os = (l->pool_size-vec3i::one) * l->in_sparse + os;
            l->my_size = os;
            os = (l->filter_size-vec3i::one) * l->in_sparse + os;
            l->in_size = os;

        }

        for ( auto& l: layers_ )
        {
            std::cout << "filter_size: " << l->filter_size << '\n'
                      << "pool_size:   " << l->pool_size << '\n'
                      << "in_size:     " << l->in_size << '\n'
                      << "my_size:     " << l->my_size << '\n'
                      << "out_size:    " << l->out_size << '\n'
                      << "in_sparse:   " << l->in_sparse << '\n'
                      << "out_sparse:  " << l->out_sparse << '\n' << "\n";
        }

        return os;
    }

    void forward( std::vector<std::vector<cube_p<real>>>& in )
    {
        std::vector<std::vector<cube_p<real>>> out;
        zi::wall_timer wt; wt.reset();
        for ( auto& l: layers_ )
        {
            std::cout << "start processing a layer" << std::endl;
            l->process_forward(in, out, tm);
            std::swap(in, out);
            std::cout << "done processing a layer: " << wt.elapsed<real>() << std::endl;
        }
    }

    void warmup()
    {
        for ( auto& l: layers_ ) { l->warmup(); }
        //for ( auto& l: layers_ ) { l->prepare_fwd(tm); }
    }
};


}}} // namespace znn::v4::trivial_fft_network
