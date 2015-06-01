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
#include "../../types.hpp"
#include "../bias.hpp"
#include "../filter.hpp"
#include "utils.hpp"

#include <map>
#include <string>
#include <vector>

namespace znn { namespace v4 { namespace trivial_fft_network {

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

    // receive a gradient for the i-th output
    // gradient is absorbed
    virtual void backward(size_t, cube_p<real>&&)
    { UNIMPLEMENTED(); }

    // receive a featuremap for the i-th input
    // featuremap is absorbed
    virtual void forward(size_t, size_t, cube_p<complex>&&)
    { UNIMPLEMENTED(); }

    // receive a gradient for the i-th output
    // gradient is absorbed
    virtual void backward(size_t, size_t, cube_p<complex>&&)
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

    virtual void set_eta( real )
    { UNIMPLEMENTED(); }

    virtual void set_momentum( real )
    { UNIMPLEMENTED(); }

    virtual void set_weight_decay( real )
    { UNIMPLEMENTED(); }

    virtual options serialize() const = 0;

};



class edge
{
public:
    virtual ~edge() {}

    // perform forward computation
    // can't modify the featuremap
    virtual void forward( ccube_p<real> const & )
    { UNIMPLEMENTED(); }

    // perform forward computation
    // can't modify the gradient
    virtual void backward( ccube_p<real> const & )
    { UNIMPLEMENTED(); }

    virtual void forward( ccube_p<complex> const & )
    { UNIMPLEMENTED(); }

    virtual void backward( ccube_p<complex> const & )
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

    void backward( ccube_p<real> const & g ) override
    {
        in_nodes->backward(in_num, impl.backward(g));
    }
};

struct dummy_edge
{
    cube_p<real> forward( ccube_p<real> const & f )
    {
        return get_copy(*f);
    }

    cube_p<real> backward( ccube_p<real> const & g )
    {
        return get_copy(*g);
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

    cube_p<real> backward( ccube_p<real> const & g )
    {
        ZI_ASSERT(indices);
        ZI_ASSERT(insize==size(*g)+(filter_size-vec3i::one)*filter_stride);

        return pooling_backprop(insize, *g, *indices);
    }
};


class real_pooling_edge
{
private:
    vec3i filter_size;

    cube_p<int> indices;
    vec3i       insize ;
    vec3i       outsize;

public:
    real_pooling_edge( vec3i const & size )
        : filter_size(size)
    {
    }

    cube_p<real> forward( ccube_p<real> const & f )
    {
        insize = size(*f);
        auto r = pooling_filter(get_copy(*f),
                                [](real a, real b){ return a>b; },
                                filter_size,
                                vec3i::one);

        outsize = insize / filter_size;
        indices = sparse_implode_slow(*r.second,filter_size,outsize);
        return sparse_implode_slow(*r.first,filter_size,outsize);
    }

    cube_p<real> backward( ccube_p<real> const & g )
    {
        ZI_ASSERT(indices);
        ZI_ASSERT(insize == size(*g) * filter_size);

        return pooling_backprop(insize, *g, *indices);
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

    cube_p<real> backward( ccube_p<real> const & g )
    {
        ZI_ASSERT(last_input);
        auto dEdW = convolve_sparse_flipped(*last_input, *g, filter_stride);
        auto ret  = convolve_sparse_inverse(*g, filter_.W(), filter_stride);
        filter_.update(*dEdW);
        return ret;
    }
};

class filter_ds_edge
{
private:
    vec3i    filter_stride;
    vec3i    repeat_;
    filter & filter_;

    ccube_p<real> last_input;

public:
    filter_ds_edge( vec3i const & stride, vec3i const & r, filter & f )
        : filter_stride(stride), repeat_(r), filter_(f)
    {
        flatten(filter_.W(), repeat_);
    }

    cube_p<real> forward( ccube_p<real> const & f )
    {
        last_input = f;
        return convolve_sparse(*f, filter_.W(), filter_stride);
    }

    cube_p<real> backward( ccube_p<real> const & g )
    {
        ZI_ASSERT(last_input);
        auto dEdW = convolve_sparse_flipped(*last_input, *g, filter_stride);
        auto ret  = convolve_sparse_inverse(*g, filter_.W(), filter_stride);
        filter_.update(*dEdW);
        flatten(filter_.W(), repeat_);
        return ret;
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

    void backward( ccube_p<complex> const & g ) override
    {
        auto dEdW_fft = *last_input * *g;
        auto dEdW = fftw::backward(std::move(dEdW_fft), in_nodes->fsize());
        real norm = dEdW->num_elements();

        flip(*dEdW);
        // TODO(zlateski): WTH was happening with sparse_implode before
        //                 when I had to use sparse_implode_slow
        //                 ony happened on my laptop
        dEdW = sparse_implode_slow(*dEdW, filter_stride, size(filter_.W()));
        *dEdW /= norm;

        auto grad = *w_fft * *g;
        filter_.update(*dEdW);

        in_nodes->backward(in_num, bwd_bucket_, std::move(grad));
    }
};


class fft_filter_ds_edge: public edge_base
{
private:
    vec3i    filter_stride;
    vec3i    repeat_;
    filter & filter_;

    ccube_p<complex> w_fft;
    ccube_p<complex> last_input;

    size_t fwd_bucket_;
    size_t bwd_bucket_;

public:
    fft_filter_ds_edge( nodes* in,
                        size_t inn,
                        nodes* out,
                        size_t outn,
                        vec3i const & stride,
                        vec3i const & r,
                        filter & flt )
        : edge_base(in, inn, out, outn)
        , filter_stride(stride)
        , repeat_(r)
        , filter_(flt)
    {
        // attach myself
        bwd_bucket_ = in->attach_out_fft_edge(inn, this);
        fwd_bucket_ = out->attach_in_fft_edge(outn, this, in->fsize());
        //flatten(filter_.W(), repeat_);
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

    void backward( ccube_p<complex> const & g ) override
    {
        auto dEdW_fft = *last_input * *g;
        auto dEdW = fftw::backward(std::move(dEdW_fft), in_nodes->fsize());
        real norm = dEdW->num_elements();

        flip(*dEdW);
        // TODO(zlateski): WTH was happening with sparse_implode before
        //                 when I had to use sparse_implode_slow
        //                 ony happened on my laptop
        dEdW = sparse_implode_slow(*dEdW, filter_stride, size(filter_.W()));
        *dEdW /= norm;

        auto grad = *w_fft * *g;
        filter_.update(*dEdW);
        flatten(filter_.W(), repeat_);

        in_nodes->backward(in_num, bwd_bucket_, std::move(grad));
    }
};



class edges
{
public:
    virtual ~edges() {}

    virtual void set_eta( real )
    { UNIMPLEMENTED(); }

    virtual void set_momentum( real )
    { UNIMPLEMENTED(); }

    virtual void set_weight_decay( real )
    { UNIMPLEMENTED(); }

    virtual options serialize() const = 0;
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

    void set_eta( real ) {}
    void set_momentum( real ) {}
    void set_weight_decay( real ) {}

    options serialize() const
    {
        return options_;
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

    void set_eta( real ) {}
    void set_momentum( real ) {}
    void set_weight_decay( real ) {}

    options serialize() const
    {
        return options_;
    }
};

class real_pooling_edges: public edges
{
private:
    options                                    options_;
    std::vector<std::unique_ptr<edge>> edges_  ;

public:
    real_pooling_edges( nodes * in,
                        nodes * out,
                        options const & opts )
        : options_(opts)
    {
        ZI_ASSERT(in->num_out_nodes()==out->num_in_nodes());

        size_t n = in->num_out_nodes();
        edges_.resize(n);

        auto sz     = opts.require_as<ovec3i>("size");

        for ( size_t i = 0; i < n; ++i )
        {
            edges_[i]
                = std::make_unique<edge_of<real_pooling_edge>>
                (in, i, out, i, sz);
        }
    }

    void set_eta( real ) {}
    void set_momentum( real ) {}
    void set_weight_decay( real ) {}

    options serialize() const
    {
        return options_;
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
        auto   repeat = opts.optional_as<ovec3i>("repeat", "1,1,1");

        size_ = sz;

        for ( size_t k = 0; k < n*m; ++k )
        {
            filters_[k] = std::make_unique<filter>(sz, eta, mom, wd);
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

            filter_values = std::string( reinterpret_cast<char*>(filters_raw),
                                         sizeof(real) * n_values );
            delete [] filters_raw;
        }

        load_filters(filters_, size_, filter_values);


        for ( size_t i = 0, k = 0; i < n; ++i )
        {
            for ( size_t j = 0; j < m; ++j, ++k )
            {
                filters_[k] = std::make_unique<filter>(sz, eta, mom, wd);
                if ( repeat == ovec3i::one )
                {
                    edges_[k]
                        = std::make_unique<edge_of<filter_edge>>
                        (in, i, out, j, stride, *filters_[k]);
                }
                else
                {
                    edges_[k]
                        = std::make_unique<edge_of<filter_ds_edge>>
                        (in, i, out, j, stride, repeat, *filters_[k]);
                }
            }
        }


    }

    void set_eta( real eta ) override
    {
        options_.push("eta", eta);
        for ( auto & f: filters_ ) f->eta() = eta;
    }

    void set_momentum( real mom ) override
    {
        options_.push("momentum", mom);
        for ( auto & f: filters_ ) f->momentum() = mom;
    }

    void set_weight_decay( real wd ) override
    {
        options_.push("weight_decay", wd);
        for ( auto & f: filters_ ) f->weight_decay() = wd;
    }

    options serialize() const
    {
        options ret = options_;
        ret.push("filters", save_filters(filters_, size_));
        return ret;
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
        auto   repeat = opts.optional_as<ovec3i>("repeat", "1,1,1");

        size_ = sz;

        for ( size_t i = 0, k = 0; i < n; ++i )
        {
            for ( size_t j = 0; j < m; ++j, ++k )
            {
                filters_[k] = std::make_unique<filter>(sz, eta, mom, wd);
                if ( repeat == ovec3i::one )
                {
                    edges_[k]
                        = std::make_unique<fft_filter_edge>
                        (in, i, out, j, stride, *filters_[k]);
                }
                else
                {
                    edges_[k]
                        = std::make_unique<fft_filter_ds_edge>
                        (in, i, out, j, stride, repeat, *filters_[k]);
                }
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

    void set_eta( real eta ) override
    {
        options_.push("eta", eta);
        for ( auto & f: filters_ ) f->eta() = eta;
    }

    void set_momentum( real mom ) override
    {
        options_.push("momentum", mom);
        for ( auto & f: filters_ ) f->momentum() = mom;
    }

    void set_weight_decay( real wd ) override
    {
        options_.push("weight_decay", wd);
        for ( auto & f: filters_ ) f->weight_decay() = wd;
    }

    options serialize() const
    {
        options ret = options_;
        ret.push("filters", save_filters(filters_, size_));
        return ret;
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

    void backward(size_t, cube_p<real>&&) override
    {
    }

    void backward(size_t, size_t, cube_p<complex>&&) override
    {
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

    void set_eta( real ) override {}
    void set_momentum( real ) override {}
    void set_weight_decay( real ) override {}

    options serialize() const
    {
        return opts_;
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

    void set_eta( real ) override {}
    void set_momentum( real ) override {}
    void set_weight_decay( real ) override {}

    options serialize() const
    {
        return opts_;
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

    void backward(size_t n, cube_p<real>&& g) override
    {
        ZI_ASSERT(n<size_);
        if ( bwd_accumulators_[n]->required() == 0 )
        {
            gs_[n] = g;
        }
        else
        {
            if ( bwd_accumulators_[n]->add(std::move(g)) )
            {
                gs_[n] = bwd_accumulators_[n]->reset();
            }
        }

        if ( gs_[n] )
        {
            bwd_dispatch_.dispatch(n,gs_[n]);
            gs_[n].reset();
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

    void backward(size_t n, size_t b, cube_p<complex>&& g) override
    {
        ZI_ASSERT(n<size_);
        if ( bwd_accumulators_[n]->add(b,std::move(g)) )
        {
            gs_[n] = bwd_accumulators_[n]->reset();
            bwd_dispatch_.dispatch(n,gs_[n]);
            gs_[n].reset();
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

    void set_eta( real eta ) override
    {
        options_.push("eta", eta);
        for ( auto& b: biases_ ) b->eta() = eta;
    }

    void set_momentum( real mom ) override
    {
        options_.push("momentum", mom);
        for ( auto& b: biases_ ) b->momentum() = mom;
    }

    void set_weight_decay( real wd ) override
    {
        options_.push("weight_decay", wd);
        for ( auto& b: biases_ ) b->weight_decay() = wd;
    }

    options serialize() const
    {
        options ret = options_;
        ret.push("biases", save_biases(biases_));
        return ret;
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

    void backward(size_t n, cube_p<real>&& g) override
    {
        ZI_ASSERT(n<size_);
        if ( bwd_accumulators_[n]->required() == 0 )
        {
            gs_[n] = g;
        }
        else
        {
            if ( bwd_accumulators_[n]->add(std::move(g)) )
            {
                gs_[n] = bwd_accumulators_[n]->reset();
            }
        }

        if ( gs_[n] )
        {
            func_.apply_grad(*gs_[n], *fs_[n]);
            biases_[n]->update(sum(*gs_[n]));
            bwd_dispatch_.dispatch(n,gs_[n]);
            gs_[n].reset();
            fs_[n].reset();
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

    void backward(size_t n, size_t b, cube_p<complex>&& g) override
    {
        ZI_ASSERT(n<size_);
        if ( bwd_accumulators_[n]->add(b,std::move(g)) )
        {
            gs_[n] = bwd_accumulators_[n]->reset();
            func_.apply_grad(*gs_[n], *fs_[n]);
            biases_[n]->update(sum(*gs_[n]));
            bwd_dispatch_.dispatch(n,gs_[n]);
            gs_[n].reset();
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
            nodes * in  = e.second->in->dnodes.get();
            nodes * out = e.second->out->dnodes.get();

            if ( type == "max_filter" )
            {
                e.second->dedges = std::make_unique<max_pooling_edges>
                    ( in, out, *e.second->opts, e.second->in_stride );
            }
            else if ( type == "max_pool" )
            {
                e.second->dedges = std::make_unique<real_pooling_edges>
                    ( in, out, *e.second->opts );
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
        else if ( type == "max_pool" )
        {
            es->width  = op.require_as<ovec3i>("size");
            es->pool   = true;
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

    void set_eta( real eta )
    {
        for ( auto & e: edges_ ) e.second->dedges->set_eta(eta);
        for ( auto & n: nodes_ ) n.second->dnodes->set_eta(eta);
    }

    void set_momentum( real mom )
    {
        for ( auto & e: edges_ ) e.second->dedges->set_momentum(mom);
        for ( auto & n: nodes_ ) n.second->dnodes->set_momentum(mom);
    }

    void set_weight_decay( real wd )
    {
        for ( auto & e: edges_ ) e.second->dedges->set_weight_decay(wd);
        for ( auto & n: nodes_ ) n.second->dnodes->set_weight_decay(wd);
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
            ret[l.first].resize(0);
        }

        return ret;
    }

    std::pair<std::vector<options>,std::vector<options>> serialize() const
    {
        std::pair<std::vector<options>,std::vector<options>> ret;

        for ( auto & n: nodes_ )
            ret.first.push_back(n.second->dnodes->serialize());

        for ( auto & e: edges_ )
            ret.second.push_back(e.second->dedges->serialize());

        return ret;
    }

    void zap()
    {
    }

};

}}} // namespace znn::v4::trivial_fft_network
