#pragma once

#include "../../assert.hpp"
#include "../../cube/cube.hpp"
#include "../../cube/cube_operators.hpp"
#include "../../options/options.hpp"
#include "../../convolution/convolution.hpp"
#include "../../transfer_function/transfer_functions.hpp"
#include "../../types.hpp"
#include "../bias.hpp"
#include "../filter.hpp"
#include "utils.hpp"

#include <map>
#include <string>
#include <vector>

namespace znn { namespace v4 {

// Forward definition
class trivial_edge;

class trivial_nodes
{
public:
    virtual ~trivial_nodes();

    // receive a featuremap for the i-th input
    // featuremap is absorbed
    virtual void forward(size_t, cube_p<double>&&)
    { UNIMPLEMENTED(); }

    // receive a gradient for the i-th output
    // gradient is absorbed
    virtual void backward(size_t, cube_p<double>&&)
    { UNIMPLEMENTED(); }

    virtual std::vector<cube_p<double>>& get_featuremaps()
    { UNIMPLEMENTED(); }

    virtual size_t num_out_nodes()
    { UNIMPLEMENTED(); }

    virtual size_t num_in_nodes()
    { UNIMPLEMENTED(); }

    virtual void attach_out_edge(size_t, trivial_edge*)
    { UNIMPLEMENTED(); }

    virtual void attach_in_edge(size_t, trivial_edge*)
    { UNIMPLEMENTED(); }

    virtual options serialize() const = 0;

};

class trivial_edge
{
protected:
    trivial_nodes * in_nodes;
    size_t          in_num;
    trivial_nodes * out_nodes;
    size_t          out_num;

public:
    trivial_edge( trivial_nodes* in,
                  size_t inn,
                  trivial_nodes* out,
                  size_t outn )
        : in_nodes(in)
        , in_num(inn)
        , out_nodes(out)
        , out_num(outn)
    {
    }

    virtual ~trivial_edge();

    // perform forward computation
    // can't modify the featuremap
    virtual void forward( ccube_p<double> const & );

    // perform forward computation
    // can't modify the gradient
    virtual void backward( ccube_p<double> const & );
};

template< typename E >
class trivial_edge_of: public trivial_edge
{
private:
    E impl;

public:
    template<class... Args>
    trivial_edge_of( trivial_nodes* in,
                     size_t inn,
                     trivial_nodes* out,
                     size_t outn,
                     Args&&... args )
        : trivial_edge(in, inn, out, outn)
        , impl(std::forward<Args>(args)...)
    {
        // attach myself
        in->attach_out_edge(inn, this);
        out->attach_in_edge(outn, this);
    }

    ~trivial_edge_of() override {}

    void forward( ccube_p<double> const & f ) override
    {
        out_nodes->forward(out_num, impl.forward(f));
    }

    void backward( ccube_p<double> const & g ) override
    {
        in_nodes->backward(in_num, impl.backward(g));
    }
};

struct trivial_dummy_edge
{
    cube_p<double> forward( ccube_p<double> const & f )
    {
        return get_copy(*f);
    }

    cube_p<double> backward( ccube_p<double> const & g )
    {
        return get_copy(*g);
    }
};

class trivial_max_pooling_edge
{
private:
    vec3i filter_size;
    vec3i filter_stride;

    cube_p<int> indices;
    vec3i       insize ;

public:
    trivial_max_pooling_edge( vec3i const & size,
                              vec3i const & stride )
        : filter_size(size), filter_stride(stride)
    {
    }

    cube_p<double> forward( ccube_p<double> const & f )
    {
        insize = size(*f);
        auto r = pooling_filter(get_copy(*f),
                                [](double a, double b){ return a>b; },
                                filter_size,
                                filter_stride);
        indices = r.second;
        return r.first;
    }

    cube_p<double> backward( ccube_p<double> const & g )
    {
        ZI_ASSERT(indices);
        ZI_ASSERT(insize == size(*g) + (filter_size - vec3i::one) * filter_stride);

        return pooling_backprop(insize, *g, *indices);
    }
};


class trivial_filter_edge
{
private:
    vec3i    filter_stride;
    filter * filter_;

    ccube_p<double> last_input;

public:
    trivial_filter_edge( vec3i const & stride, filter * f )
        : filter_stride(stride), filter_(f)
    {
    }

    cube_p<double> forward( ccube_p<double> const & f )
    {
        last_input = f;
        return convolve_sparse(*f, filter_->W(), filter_stride);
    }

    cube_p<double> backward( ccube_p<double> const & g )
    {
        ZI_ASSERT(last_input);
        auto dEdW = convolve_sparse_flipped(*last_input, *g, filter_stride);
        auto ret  = convolve_sparse_inverse(*g, filter_->W(), filter_stride);
        filter_->update(*dEdW);
        return ret;
    }
};


class trivial_input_nodes: public trivial_nodes
{
private:
    size_t                                  size_   ;
    std::vector<std::vector<trivial_edge*>> outputs_;

public:
    trivial_input_nodes(size_t s)
        : size_(s)
        , outputs_(s)
    {}

    void forward(size_t n, cube_p<double>&& f) override
    {
        ZI_ASSERT(n<size_);
        for ( auto& e: outputs_[n] )
        {
            e->forward(f);
        }
    }

    void backward(size_t, cube_p<double>&&) override
    {
    }

    size_t num_out_nodes() override { return size_; }
    size_t num_in_nodes()  override { return size_; }

    void attach_out_edge(size_t i, trivial_edge* e) override
    {
        ZI_ASSERT(i<size_);
        outputs_[i].push_back(e);
    }

    options serialize() const
    {
        options ret;
        ret.push("type", "input").push("size", size_);
        return ret;
    }
};


class trivial_summing_nodes: public trivial_nodes
{
private:
    size_t                                  size_    ;
    std::vector<std::vector<trivial_edge*>> inputs_  ;
    std::vector<std::vector<trivial_edge*>> outputs_ ;
    std::vector<size_t>                     received_;
    std::vector<cube_p<double>>             fs_      ;
    std::vector<cube_p<double>>             gs_      ;

public:
    trivial_summing_nodes(size_t s)
        : size_(s)
        , inputs_(s)
        , outputs_(s)
        , received_(s)
        , fs_(s)
        , gs_(s)
    {}

    options serialize() const
    {
        options ret;
        ret.push("type", "sum").push("size", size_);
        return ret;
    }

    std::vector<cube_p<double>>& get_featuremaps() override
    {
        return fs_;
    }

    void forward(size_t n, cube_p<double>&& f) override
    {
        ZI_ASSERT(n<size_);
        if ( received_[n] == 0 )
        {
            fs_[n] = f;
        }
        else
        {
            *fs_[n] += *f;
        }

        if ( ++received_[n] == outputs_[n].size() )
        {
            for ( auto& e: outputs_[n] )
            {
                e->forward(fs_[n]);
            }
            received_[n] = 0;
            fs_[n].reset();
        }
    }

    void backward(size_t n, cube_p<double>&& g) override
    {
        ZI_ASSERT(n<size_);
        if ( received_[n] == 0 )
        {
            gs_[n] = g;
        }
        else
        {
            *gs_[n] += *g;
        }

        if ( ++received_[n] == inputs_[n].size() )
        {
            for ( auto& e: inputs_[n] )
            {
                e->backward(gs_[n]);
            }
            received_[n] = 0;
            gs_[n].reset();
        }
    }

    size_t num_out_nodes() override { return size_; }
    size_t num_in_nodes()  override { return size_; }

    void attach_in_edge(size_t i, trivial_edge* e) override
    {
        ZI_ASSERT(i<size_);
        inputs_[i].push_back(e);
    }

    void attach_out_edge(size_t i, trivial_edge* e) override
    {
        ZI_ASSERT(i<size_);
        outputs_[i].push_back(e);
    }
};


template<class F>
class trivial_transfer_nodes: public trivial_nodes
{
private:
    size_t                                  size_    ;
    std::vector<std::unique_ptr<bias>>&     biases_  ;
    transfer_function_wrapper<F>            func_    ;
    std::vector<std::vector<trivial_edge*>> inputs_  ;
    std::vector<std::vector<trivial_edge*>> outputs_ ;
    std::vector<size_t>                     received_;
    std::vector<cube_p<double>>             fs_      ;
    std::vector<cube_p<double>>             gs_      ;

public:
    trivial_transfer_nodes(size_t s, std::vector<std::unique_ptr<bias>>& bs,
                           const F& fn = F() )
        : size_(s)
        , biases_(bs)
        , func_(fn)
        , inputs_(s)
        , outputs_(s)
        , received_(s)
        , fs_(s)
        , gs_(s)
    {}

    options serialize() const
    {
        options ret;
        ret.push("type", "transfer").
            push("size", size_).
            push(func_.serialize()).
            push("biases", save_biases(biases_));
        return ret;
    }

    std::vector<cube_p<double>>& get_featuremaps() override
    {
        return fs_;
    }

    void forward(size_t n, cube_p<double>&& f) override
    {
        ZI_ASSERT(n<size_);
        if ( received_[n] == 0 )
        {
            fs_[n] = f;
        }
        else
        {
            *fs_[n] += *f;
        }

        if ( ++received_[n] == outputs_[n].size() )
        {
            func_.apply(fs_[n], biases_[n]->b());
            for ( auto& e: outputs_[n] )
            {
                e->forward(fs_[n]);
            }
            received_[n] = 0;
        }
    }

    void backward(size_t n, cube_p<double>&& g) override
    {
        ZI_ASSERT(n<size_);
        if ( received_[n] == 0 )
        {
            gs_[n] = g;
        }
        else
        {
            *gs_[n] += *g;
        }

        if ( ++received_[n] == inputs_[n].size() )
        {
            func_.apply_grad(gs_[n], fs_[n]);
            biases_[n]->update(sum(*gs_[n]));

            for ( auto& e: inputs_[n] )
            {
                e->backward(gs_[n]);
            }

            received_[n] = 0;
            gs_[n].reset();
            fs_[n].reset();
        }
    }

    size_t num_out_nodes() override { return size_; }
    size_t num_in_nodes()  override { return size_; }

    void attach_in_edge(size_t i, trivial_edge* e) override
    {
        ZI_ASSERT(i<size_);
        inputs_[i].push_back(e);
    }

    void attach_out_edge(size_t i, trivial_edge* e) override
    {
        ZI_ASSERT(i<size_);
        outputs_[i].push_back(e);
    }
};




}} // namespace znn::v4
