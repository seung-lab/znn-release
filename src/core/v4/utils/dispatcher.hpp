#pragma once

#include "../types.hpp"
#include "../cube/cube_operators.hpp"
#include "../fft/fftw.hpp"
#include "task_manager.hpp"

#include <zi/utility/non_copyable.hpp>
#include <vector>
#include <memory>

namespace znn { namespace v4 {

template<class Edge, class FFTEdge>
struct dispatcher_base: private zi::non_copyable
{
    typedef Edge    edge_type;
    typedef FFTEdge fft_edge_type;
};


template<class Edge, class FFTEdge>
class forward_dispatcher: public dispatcher_base<Edge, FFTEdge>
{
private:
    std::vector<Edge*>                    targets_    ;
    std::map<vec3i,std::vector<FFTEdge*>> fft_targets_;

public:
    void dispatch(const ccube_p<real>& v) const
    {
        ZI_ASSERT(fft_targets_.size()<2);
        for ( auto& t: targets_ )
        {
            t->forward(v);
        }
        for ( auto& fft_target: fft_targets_ )
        {
            ccube_p<complex> x = fftw::forward_pad(v,fft_target.first);
            for ( auto& t: fft_target.second )
            {
                t->forward(x);
            }
        }
    }

    void sign_up(Edge* e)
    {
        targets_.push_back(e);
    }

    void sign_up(const vec3i& s, FFTEdge* e)
    {
        fft_targets_[s].push_back(e);
    }

}; // class forward_dispatcher


template<class Edge, class FFTEdge>
class backward_dispatcher: public dispatcher_base<Edge, FFTEdge>
{
private:
    std::vector<Edge*>                    targets_    ;
    std::map<vec3i,std::vector<FFTEdge*>> fft_targets_;

public:
    void dispatch(const ccube_p<real>& v) const
    {
        for ( auto& t: targets_ )
        {
            t->backward(v);
        }

        cube_p<real> vp = get_copy(*v);
        flip(*vp);

        for ( auto& fft_target: fft_targets_ )
        {
            ccube_p<complex> x = fftw::forward_pad(vp,fft_target.first);
            for ( auto& t: fft_target.second )
            {
                t->backward(x);
            }
        }
    }

    void sign_up(Edge* e)
    {
        targets_.push_back(e);
    }

    void sign_up(const vec3i& s, FFTEdge* e)
    {
        fft_targets_[s].push_back(e);
    }

}; // class backward_dispatcher


template<class Edge, class FFTEdge>
class concurrent_forward_dispatcher: public dispatcher_base<Edge, FFTEdge>
{
private:
    std::vector<Edge*>                    targets_    ;
    std::map<vec3i,std::vector<FFTEdge*>> fft_targets_;

    typedef concurrent_forward_dispatcher this_type;

private:
    void fft_dispatch( ccube_p<real> const & v,
                       vec3i const & s,
                       std::vector<FFTEdge*> const & targets,
                       task_manager & manager ) const
    {
        ccube_p<complex> x = fftw::forward_pad(v, s);
        for ( auto& t: targets )
        {
            manager.asap([t,x](){t->forward(x);});
        }
    }

public:
    void dispatch( ccube_p<real> const & v,
                   task_manager & manager) const
    {
        for ( auto& t: targets_ )
            manager.asap([t,v](){t->forward(v);});

        for ( auto& fft_target: fft_targets_ )
            manager.asap(&this_type::fft_dispatch,this,v,fft_target.first,
                         std::cref(fft_target.second), std::ref(manager));
    }

    void sign_up(Edge* e)
    {
        targets_.push_back(e);
    }

    void sign_up(const vec3i& s, FFTEdge* e)
    {
        fft_targets_[s].push_back(e);
    }

}; // class concurrent_forward_dispatcher



template<class Edge, class FFTEdge>
class concurrent_backward_dispatcher: public dispatcher_base<Edge, FFTEdge>
{
private:
    std::vector<Edge*>                    targets_    ;
    std::map<vec3i,std::vector<FFTEdge*>> fft_targets_;

    typedef concurrent_backward_dispatcher this_type;

private:
    void fft_dispatch( const ccube_p<real>& v, const vec3i& s,
                       const std::vector<FFTEdge*>& targets,
                       task_manager& manager ) const
    {
        auto vp = get_copy(*v);
        flip(*vp);

        ccube_p<complex> x = fftw::forward_pad(std::move(vp), s);

        for ( auto& t: targets )
        {
            manager.asap([t,x](){t->backward(x);});
        }
    }

public:
    void dispatch(const ccube_p<real>& v, task_manager& manager) const
    {
        for ( auto& t: targets_ )
            manager.asap([t,v](){t->backward(v);});

        for ( auto& fft_target: fft_targets_ )
            manager.asap(&this_type::fft_dispatch,this,v,fft_target.first,
                         std::cref(fft_target.second), std::ref(manager));
    }

    void sign_up(Edge* e)
    {
        targets_.push_back(e);
    }

    void sign_up(const vec3i& s, FFTEdge* e)
    {
        fft_targets_[s].push_back(e);
    }

}; // class concurrent_backward_dispatcher


template<typename Dispatcher>
class dispatcher_group
{
private:
    typedef typename Dispatcher::edge_type     edge_type    ;
    typedef typename Dispatcher::fft_edge_type fft_edge_type;

private:
    std::vector<Dispatcher> dispatchers_;

public:
    dispatcher_group(size_t s)
        : dispatchers_(s)
    {}

    void dispatch(size_t i, const ccube_p<real>& v) const
    {
        ZI_ASSERT(i<dispatchers_.size());
        dispatchers_[i].dispatch(v);
    }

    template<class M>
    void dispatch(size_t i, ccube_p<real> const & v, M & man)
    {
        ZI_ASSERT(i<dispatchers_.size());
        dispatchers_[i].dispatch(v,man);
    }

    void sign_up(size_t i, edge_type* e)
    {
        ZI_ASSERT(i<dispatchers_.size());
        dispatchers_[i].sign_up(e);
    }

    void sign_up(size_t i, const vec3i& s, fft_edge_type* e)
    {
        ZI_ASSERT(i<dispatchers_.size());
        dispatchers_[i].sign_up(s,e);
    }

};



}} // namespace znn::v4
