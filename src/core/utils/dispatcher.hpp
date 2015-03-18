//
// Copyright (C) 2015-present  Aleksandar Zlateski <zlateski@mit.edu>
// ------------------------------------------------------------------
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

#ifndef ZNN_CORE_UTILS_DISPATCHER_HPP_INCLUDED
#define ZNN_CORE_UTILS_DISPATCHER_HPP_INCLUDED

#include "../types.hpp"
#include "../utils.hpp"
#include "../volume_pool.hpp"
#include "../task_manager.hpp"
#include "../volume_operators.hpp"
#include "../fft/fftw.hpp"

#include <zi/utility/non_copyable.hpp>
#include <vector>
#include <memory>

namespace zi { namespace znn {

template<class Edge, class FFTEdge>
struct dispatcher_base: private non_copyable
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
    template<typename M>
    void dispatch(const cvol_p<double>& v, M& /* task_manager */) const
    {
        //ZI_ASSERT(fft_targets_<2);
        for ( auto& t: targets_ )
        {
            t->forward(v);
        }
        for ( auto& fft_target: fft_targets_ )
        {
            cvol_p<complex> x = fftw::forward_pad(v,fft_target.first);
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
    template<typename M>
    void dispatch(const cvol_p<double>& v, M& /* task_manager */) const
    {
        for ( auto& t: targets_ )
        {
            t->backward(v);
        }

        vol_p<double> vp = get_volume<double>(size(*v));
        *vp = *v;
        flip_vol(*vp);

        for ( auto& fft_target: fft_targets_ )
        {
            cvol_p<complex> x = fftw::forward_pad(vp,fft_target.first);
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
    void fft_dispatch( const cvol_p<double>& v, const vec3i& s,
                       const std::vector<FFTEdge*>& targets,
                       task_manager& manager ) const
    {
        cvol_p<complex> x = fftw::forward_pad(v, s);
        for ( auto& t: targets )
        {
            manager.asap(&FFTEdge::forward,t,x);
        }
    }

public:
    void dispatch(const cvol_p<double>& v, task_manager& manager) const
    {
        for ( auto& t: targets_ )
            manager.asap(&Edge::forward,t,v);

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
    void fft_dispatch( const cvol_p<double>& v, const vec3i& s,
                       const std::vector<FFTEdge*>& targets,
                       task_manager& manager ) const
    {
        vol_p<double> vp = get_volume<double>(size(*v));
        *vp = *v;
        flip_vol(*vp);
        cvol_p<complex> x = fftw::forward_pad(std::move(vp), s);

        for ( auto& t: targets )
        {
            manager.asap(&FFTEdge::backward,t,x);
        }
    }

public:
    void dispatch(const cvol_p<double>& v, task_manager& manager) const
    {
        for ( auto& t: targets_ )
            manager.asap(&Edge::backward,t,v);

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

    void dispatch(size_t i, const cvol_p<double>& v, task_manager& m) const
    {
        ZI_ASSERT(i<dispatchers_.size());
        dispatchers_[i].dispatch(v,m);
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



}} // namespace zi::znn

#endif // ZNN_CORE_UTILS_DISPATCHER_HPP_INCLUDED
