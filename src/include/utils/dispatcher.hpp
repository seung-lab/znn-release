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
    std::vector<Edge*>                           targets_    ;
    std::map<vec3i,std::vector<FFTEdge*>>        fft_targets_;
    std::map<vec3i,std::unique_ptr<fftw::transformer>>  fftw_;

public:
    void dispatch(const ccube_p<real>& v)
    {
        ZI_ASSERT(fft_targets_.size()<2);
        for ( auto& t: targets_ )
            t->forward(v);

        for ( auto& fft_target: fft_targets_ )
        {
            ccube_p<complex> x = fftw_[fft_target.first]->forward_pad(v);
            for ( auto& t: fft_target.second )
                t->forward(x);
        }
    }

    void sign_up(Edge* e)
    {
        targets_.push_back(e);
    }

    void sign_up(const vec3i& s, FFTEdge* e)
    {
        if ( fftw_.count(s) == 0 )
        {
            fftw_[s] = std::make_unique<fftw::transformer>(s);
        }

        fft_targets_[s].push_back(e);
    }

    void enable(bool b)
    {
        for ( auto& t: targets_ )
            t->enable_fwd(b);

        for ( auto& fft_target: fft_targets_)
            for ( auto& t: fft_target.second )
                t->enable_fwd(b);
    }

    size_t size() const
    {
        size_t r = targets_.size();
        for ( auto& fft_target: fft_targets_)
            r += fft_target.second.size();
        return r;
    }

}; // class forward_dispatcher


template<class Edge, class FFTEdge>
class backward_dispatcher: public dispatcher_base<Edge, FFTEdge>
{
private:
    std::vector<Edge*>                           targets_    ;
    std::map<vec3i,std::vector<FFTEdge*>>        fft_targets_;
    std::map<vec3i,std::unique_ptr<fftw::transformer>>  fftw_;

public:
    void dispatch(const ccube_p<real>& v)
    {
        for ( auto& t: targets_ )
            t->backward(v);

        cube_p<real> vp = get_copy(*v);
        flip(*vp);

        for ( auto& fft_target: fft_targets_ )
        {
            ccube_p<complex> x = fftw_[fft_target.first]->forward_pad(vp);
            for ( auto& t: fft_target.second )
                t->backward(x);
        }
    }

    void sign_up(Edge* e)
    {
        targets_.push_back(e);
    }

    void sign_up(const vec3i& s, FFTEdge* e)
    {
        if ( fftw_.count(s) == 0 )
        {
            fftw_[s] = std::make_unique<fftw::transformer>(s);
        }

        fft_targets_[s].push_back(e);
    }

    void enable(bool b)
    {
        for ( auto& t: targets_ )
            t->enable_bwd(b);

        for ( auto& fft_target: fft_targets_)
            for ( auto& t: fft_target.second )
                t->enable_bwd(b);
    }

    size_t size() const
    {
        size_t r = targets_.size();
        for ( auto& fft_target: fft_targets_)
            r += fft_target.second.size();
        return r;
    }

}; // class backward_dispatcher


template<class Edge, class FFTEdge>
class concurrent_forward_dispatcher: public dispatcher_base<Edge, FFTEdge>
{
private:
    std::vector<Edge*>                           targets_    ;
    std::map<vec3i,std::vector<FFTEdge*>>        fft_targets_;
    std::map<vec3i,std::unique_ptr<fftw::transformer>>  fftw_;

    typedef concurrent_forward_dispatcher this_type;

private:
    void fft_dispatch( ccube_p<real> const & v,
                       vec3i const & s,
                       std::vector<FFTEdge*> const & targets,
                       task_manager & manager )
    {
        ccube_p<complex> x = fftw_[s]->forward_pad(v);
        for ( auto& t: targets )
        {
            manager.schedule(t->fwd_priority(), [t,x](){t->forward(x);});
        }
    }

public:
    void dispatch( ccube_p<real> const & v,
                   task_manager & manager)
    {
        for ( auto& t: targets_ )
            manager.schedule(t->fwd_priority(), [t,v](){t->forward(v);});

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
        if ( fftw_.count(s) == 0 )
        {
            fftw_[s] = std::make_unique<fftw::transformer>(s);
        }

        fft_targets_[s].push_back(e);
    }

    void enable(bool b)
    {
        for ( auto& t: targets_ )
            t->enable_fwd(b);

        for ( auto& fft_target: fft_targets_)
            for ( auto& t: fft_target.second )
                t->enable_fwd(b);
    }

    size_t size() const
    {
        size_t r = targets_.size();
        for ( auto& fft_target: fft_targets_)
            r += fft_target.second.size();
        return r;
    }

}; // class concurrent_forward_dispatcher



template<class Edge, class FFTEdge>
class concurrent_backward_dispatcher: public dispatcher_base<Edge, FFTEdge>
{
private:
    std::vector<Edge*>                           targets_    ;
    std::map<vec3i,std::vector<FFTEdge*>>        fft_targets_;
    std::map<vec3i,std::unique_ptr<fftw::transformer>>  fftw_;

    typedef concurrent_backward_dispatcher this_type;

private:
    void fft_dispatch( const ccube_p<real>& v, const vec3i& s,
                       const std::vector<FFTEdge*>& targets,
                       task_manager& manager )
    {
        auto vp = get_copy(*v);
        flip(*vp);

        ccube_p<complex> x = fftw_[s]->forward_pad(std::move(vp));

        for ( auto& t: targets )
        {
            manager.schedule(t->bwd_priority(), [t,x](){t->backward(x);});
        }
    }

public:
    void dispatch(const ccube_p<real>& v, task_manager& manager)
    {
        for ( auto& t: targets_ )
            manager.schedule(t->bwd_priority(), [t,v](){t->backward(v);});

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
        if ( fftw_.count(s) == 0 )
        {
            fftw_[s] = std::make_unique<fftw::transformer>(s);
        }

        fft_targets_[s].push_back(e);
    }

    void enable(bool b)
    {
        for ( auto& t: targets_ )
            t->enable_bwd(b);

        for ( auto& fft_target: fft_targets_)
            for ( auto& t: fft_target.second )
                t->enable_bwd(b);
    }

    size_t size() const
    {
        size_t r = targets_.size();
        for ( auto& fft_target: fft_targets_)
            r += fft_target.second.size();
        return r;
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

    void dispatch(size_t i, const ccube_p<real>& v)
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

    void enable(size_t i, bool b)
    {
        ZI_ASSERT(i<dispatchers_.size());
        dispatchers_[i].enable(b);
    }

    size_t size(size_t i) const
    {
        ZI_ASSERT(i<dispatchers_.size());
        return dispatchers_[i].size();
    }

};



}} // namespace znn::v4
