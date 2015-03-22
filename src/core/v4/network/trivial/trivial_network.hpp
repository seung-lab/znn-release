#pragma once

#include "../../transfer_function/transfer_functions.hpp"
#include "../../types.hpp"
#include "../../cube/cube.hpp"
#include "../../cube/cube_operators.hpp"
#include "../../assert.hpp"
#include "../bias.hpp"
#include "../filter.hpp"
#include "../../options/options.hpp"
#include "utils.hpp"

#include <vector>
#include <map>
#include <string>


namespace znn { namespace v4 {

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
public:
    virtual ~trivial_edge();

    // perform forward computation
    // can't modify the featuremap
    virtual void forward(const ccube_p<double>&);

    // perform forward computation
    // can't modify the gradient
    virtual void backward(const ccube_p<double>&);
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
        ret.push("type", "transfer").push("size", size_);
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
