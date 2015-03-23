//
// Copyright (C) 2014  Aleksandar Zlateski <zlateski@mit.edu>
//                     Kisuk Lee           <kisuklee@mit.edu>
// ----------------------------------------------------------
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

#ifndef ZNN_NODE_HPP_INCLUDED
#define ZNN_NODE_HPP_INCLUDED

#include "types.hpp"
#include "utils.hpp"
#include "volume_pool.hpp"
#include "volume_utils.hpp"
#include "bf_conv.hpp"
#include "fftw.hpp"
#include "generic_filter.hpp"
#include "../error_fn/error_fns.hpp"

#include <stdexcept>
#include <list>

#include <zi/utility/for_each.hpp>

namespace zi {
namespace znn {

// forward declaration
class node;
class edge;

typedef boost::shared_ptr<node> node_ptr;
typedef boost::shared_ptr<edge> edge_ptr;

class edge: public zi::mutex
{
private:
    vec3i           size_  ;
    node_ptr        in_    ;
    node_ptr        out_   ;
    double3d_ptr    W_     ;
    complex3d_ptr   fft_   ;
    vec3i           sparse_;

    // norm
    double          norm_;

    // update
    double          eta_;    // learning rate parameter
    double          mom_;    // momentum parameter
    double3d_ptr    V_  ;    // momentum volume
    double          wc_ ;    // weight decay parameter
    double          szB_;    // minibatch size (for averaging gradients)


public:
    edge(node_ptr in, node_ptr out, std::size_t x, std::size_t y, std::size_t z,
         double eta, double mom = 0.0, double wc = 0.0, double szB = 1.0)
        : size_(x,y,z)
        , in_(in)
        , out_(out)
        , W_(volume_pool.get_double3d(x,y,z))
        , fft_()
        , sparse_(vec3i::one)
        , norm_(0.0)
        , eta_(eta)
        , mom_(mom)
        , V_(volume_pool.get_double3d(x,y,z))
        , wc_(wc)
        , szB_(szB)
    {
        volume_utils::zero_out(V_);
    }

    edge(node_ptr in, node_ptr out, double3d_ptr W, double eta = 0.01,
         double mom = 0.0, double wc = 0.0, double szB = 1.0)
        : size_(W->shape()[0],W->shape()[1],W->shape()[2])
        , in_(in)
        , out_(out)
        , W_(W)
        , fft_()
        , sparse_(vec3i::one)
        , norm_(0.0)
        , eta_(eta)
        , mom_(mom)
        , V_(volume_pool.get_double3d(W))
        , wc_(wc)
        , szB_(szB)
    {
        volume_utils::zero_out(V_);
    }

    void unlink()
    {
        in_.reset();
        out_.reset();
        W_.reset();
        V_.reset();
        fft_.reset();
    }

    void set_eta(double eta)
    {
        eta_ = eta;
    }

    void set_W( double3d_ptr W, vec3i s = vec3i::one )
    {
        ASSERT_SAME_SIZE(W_,W);
        W_ = W;
        set_sparse(s);
    }

    void set_sparse( vec3i s = vec3i::one )
    {
        if ( sparse_ != s )
        {
            sparse_ = s;

            if ( s != vec3i::one )
            {
                W_ = volume_utils::sparse_decompress(W_,sparse_);
                V_ = volume_utils::sparse_decompress(V_,sparse_);
            }
        }
    }

    void reset_W( double3d_ptr W )
    {
        if ( size_of(W) == size_ )
        {
            W_ = volume_utils::sparse_decompress(W,sparse_);
        }
        else
        {
            std::cout << "reset_W: something bad happend." << std::endl;
        }
    }

    void reset_V()
    {
        volume_utils::zero_out(V_);
    }

    void set_momentum( double mom )
    {
        mom_ = std::max(0.0, std::min(1.0, mom));
    }

    void set_weight_decay( double wc )
    {
        wc_ = std::max(0.0, std::min(1.0, wc));
    }

    void set_minibatch_size( double sz )
    {
        szB_ = std::max(1.0, sz);
    }

    vec3i get_sparse() const
    {
        return sparse_;
    }

    vec3i size() const
    {
        return size_;
    }

    vec3i real_size() const
    {
        return vec3i((size_[0]-1)*sparse_[0]+1,
                     (size_[1]-1)*sparse_[1]+1,
                     (size_[2]-1)*sparse_[2]+1);

    }

    friend class node;

// for exporting net
public:
    void print_W( std::ostream& stream )
    {
        for ( std::size_t z = 0; z < W_->shape()[2]; z += sparse_[2] )
            for ( std::size_t y = 0; y < W_->shape()[1]; y += sparse_[1] )
                for ( std::size_t x = 0; x < W_->shape()[0]; x += sparse_[0] )
                {
                    double d = (*W_)[x][y][z];
                    stream.write( reinterpret_cast<char*>(&d), sizeof(double) );
                }
    }

}; // class edge

class node
{
private:
    const std::string       name_;

    zi::mutex               m_ ;
    zi::condition_variable  cv_;

    std::size_t             pass_no_;
    std::size_t             waiters_;

    vec3i size_;

    // actual data
    double       bias_;
    double3d_ptr f_   ;
    long3d_ptr   filter_indices_;
    double3d_ptr dEdX_;
//    double       dEdB_;
    double       eta_ ;

    // fft
    complex3d_ptr fft_;
    complex3d_ptr dEdX_fft_;

    bool receives_fft_;
    bool sends_fft_;

    // links
    std::list<edge_ptr> in_edges_ ;
    std::list<edge_ptr> out_edges_;

    std::size_t in_received_;
    std::size_t out_received_;

    error_fn_ptr error_fn_;

    friend class edge;

    std::size_t layer_no_;
    std::size_t neuron_no_;

    vec3i filter_size_;
    vec3i sparse_     ;
    vec3i step_size_  ;
    vec3i real_filter_size_;
    zi::function<bool(double,double)> filter_function_;

    double mom_;
    double v_  ;
    double wc_ ;
    double szB_;

public:
    node(const std::string& name, double bias = 0.01, double eta = 0.0001,
         std::size_t layer_no = 0, std::size_t neuron_no = 0,
         error_fn_ptr error_fn = error_fn_ptr(new logistic_error_fn),
         double mom = 0.0, double wc = 0.0, double szB = 1.0)
        : name_(name)
        , m_()
        , cv_()
        , pass_no_(0)
        , waiters_(0)
        , size_()
        , bias_(bias)
        , f_()
        , filter_indices_()
        , dEdX_()
//        , dEdB_(0)
        , eta_(eta)
        , fft_()
        , dEdX_fft_()
        , receives_fft_()
        , sends_fft_()
        , in_edges_()
        , out_edges_()
        , in_received_(0)
        , out_received_(0)
        , error_fn_(error_fn)
        , layer_no_(layer_no)
        , neuron_no_(neuron_no)
        , filter_size_(vec3i::one)
        , sparse_(vec3i::one)
        , step_size_(vec3i::one)
        , real_filter_size_()
        , filter_function_()
        , mom_(mom)
        , v_(0.0)
        , wc_(wc)
        , szB_(szB)
    {
        real_filter_size_ = (filter_size_-vec3i::one)*sparse_+vec3i::one;
    }

    void set_error_fn(error_fn_ptr ef)
    {
        error_fn_ = ef;
    }

    error_fn_ptr get_error_fn() const
    {
        return error_fn_;
    }

    void set_layer_number( std::size_t l )
    {
        layer_no_ = l;
    }

    std::size_t get_layer_number() const
    {
        return layer_no_;
    }

    std::size_t forward_priority() const
    {
        return layer_no_ * 1000 + neuron_no_;
    }

    std::size_t backward_priority() const
    {
        return 2000000000 - (layer_no_ * 1000 + neuron_no_);
    }

    vec3i get_size() const
    {
        return size_;
    }

    void set_size( const vec3i& size )
    {
        size_ = size;
    }

    void init(vec3i s)
    {
        if ( size_ == vec3i::zero )
        {
            size_ = s;

            FOR_EACH( it, out_edges_ )
            {
                (*it)->out_->init(s-(*it)->real_size()-real_filter_size_+vec3i::one+vec3i::one);
            }
        }
        else
        {
            if ( size_ != s )
            {
                throw std::logic_error("Sizes aint matching!");
            }
        }
    }

    vec3i get_sparse() const
    {
        return sparse_;
    }

    void set_sparse( vec3i s )
    {
        sparse_ = s;
        real_filter_size_ = (filter_size_-vec3i::one)*sparse_+vec3i::one;
    }

    vec3i get_step_size() const
    {
        return step_size_;
    }

    void set_step_size(vec3i s)
    {
        step_size_ = s;
    }

    void reset()
    {
        if ( size_ != vec3i::zero )
        {
            size_ = vec3i::zero;
            FOR_EACH( it, out_edges_ )
            {
                (*it)->out_->reset();
            }
        }
    }

    // for computing input size from output size
    void backward_init(vec3i s)
    {
        if ( size_ == vec3i::zero )
        {
            size_ = s + real_filter_size_ - vec3i::one;

            // std::cout << name_ << ": " << size_ << std::endl;

            FOR_EACH( it, in_edges_ )
            {
                if ( (*it)->size_ == vec3i::one )
                {
                    receives_fft_ = false;
                    (*it)->in_->sends_fft_ = false;
                }

                // const vec3i& sparse = (*it)->get_sparse();
                (*it)->in_->backward_init(s+(*it)->real_size()+real_filter_size_
                                          -vec3i::one-vec3i::one);
            }
        }
        else
        {
            if ( size_ != s + real_filter_size_ - vec3i::one )
            {
                throw std::logic_error("Sizes aint matching!");
            }
        }
    }

    void init(std::size_t x, std::size_t y, std::size_t z)
    {
        init(vec3i(x,y,z));
    }

    // for computing input size from output size
    void backward_init(std::size_t x, std::size_t y, std::size_t z)
    {
        backward_init(vec3i(x,y,z));
    }

    // setters and getters for bias and eta
    void set_bias(double bias)
    {
        bias_ = bias;
    }

    // momentum for bias
    void set_v(double v)
    {
        v_ = v;
    }

    void reset_v()
    {
        v_ = static_cast<double>(0);
    }

    double get_bias() const
    {
        return bias_;
    }

    void set_eta(double eta)
    {
        eta_ = eta;
    }

    double get_eta() const
    {
        return eta_;
    }

    // momentum
    void set_momentum( double mom )
    {
        mom_ = std::max(0.0, std::min(1.0, mom));
    }

    // weight decay
    void set_weight_decay( double wc )
    {
        wc_ = std::max(0.0, std::min(1.0, wc));
    }

    // for average gradient update
    void set_minibatch_size( double sz )
    {
        szB_ = std::max(1.0, sz);
    }

    template<typename F>
    void set_filtering(const F& f, const vec3i& s, const vec3i& sp = vec3i::one)
    {
        filter_size_ = s;
        set_sparse(sp);
        filter_function_ = f;
    }

    const vec3i& get_filter_size() const
    {
        return filter_size_;
    }

    std::string get_name() const
    {
        return name_;
    }

    std::size_t get_neuron_number() const
    {
        return neuron_no_;
    }

    std::size_t count_in_edges() const
    {
        return in_edges_.size();
    }

    std::size_t count_out_edges() const
    {
        return out_edges_.size();
    }

private:
    template<class Manager>
    void forward_edge(edge_ptr e, Manager task_manager)
    {
        zi::guard g(*e);

        if ( sends_fft_ )
        {
            ZI_ASSERT(e->out_->receives_fft_);
            if ( !e->fft_ )
            {
                e->fft_ = fftw::forward_pad(e->W_,out_size());
            }
            e->out_->template receive_f<Manager>(
                volume_utils::elementwise_mul(fft_,e->fft_), task_manager);
        }
        else
        {
            ZI_ASSERT(!e->out_->receives_fft_);
            if ( e->size_ == vec3i::one )
            {
                e->out_->template receive_f<Manager>(bf_conv_constant(f_, (*e->W_)[0][0][0]), task_manager);
            }
            else
            {
                if ( e->sparse_ == vec3i::one )
                {
                    e->out_->template receive_f<Manager>(bf_conv(f_, e->W_), task_manager);
                }
                else
                {
                    e->out_->template receive_f<Manager>(bf_conv_sparse(f_, e->W_, e->sparse_), task_manager);
                }
            }
        }
    }

    template<class Manager>
    void backward_edge(edge_ptr e, Manager task_manager)
    {
        zi::guard g(*e);
        double3d_ptr dEdW;

        if ( receives_fft_ )
        {
            ZI_ASSERT(e->in_->sends_fft_);

            complex3d_ptr dEdW_fft
                = volume_utils::elementwise_mul(e->in_->fft_, dEdX_fft_);

            vec3i s = in_edges_.front()->in_->out_size();

            dEdW = fftw::backward(std::move(dEdW_fft),s);

            // [TODO: zlateski]  normalize after cropping
            dEdW = volume_utils::normalize_flip(dEdW);
            dEdW = volume_utils::crop_left(dEdW,e->real_size());

            complex3d_ptr grad
                = volume_utils::elementwise_mul(dEdX_fft_, e->fft_);

            e->in_->template receive_grad<Manager>(grad, task_manager);
        }
        else
        {
            ZI_ASSERT(!e->in_->sends_fft_);

            if ( e->size_ == vec3i::one )
            {
                dEdW = volume_pool.get_double3d(1,1,1);
                (*dEdW)[0][0][0] = bf_conv_flipped_constant(e->in_->f_, dEdX_);
                double3d_ptr grad = bf_conv_inverse_constant(dEdX_, (*e->W_)[0][0][0]);
                e->in_->template receive_grad<Manager>(grad, task_manager);
            }
            else
            {
                if ( e->sparse_ == vec3i::one )
                {
                    dEdW = bf_conv_flipped(e->in_->f_, dEdX_);
                    double3d_ptr grad = bf_conv_inverse(dEdX_, e->W_);
                    e->in_->template receive_grad<Manager>(grad, task_manager);
                }
                else
                {
                    dEdW = bf_conv_flipped_sparse(e->in_->f_, dEdX_, e->sparse_);
                    double3d_ptr grad = bf_conv_inverse_sparse(dEdX_, e->W_, e->sparse_);
                    e->in_->template receive_grad<Manager>(grad, task_manager);
                }
            }
        }

        update_edge(e, dEdW);
    }

    void update_edge(edge_ptr e, double3d_ptr dEdW)
    {
        // update weight
        volume_utils::elementwise_mul_by(e->V_, e->mom_);
        volume_utils::elementwise_div_by(dEdW, szB_);
        volume_utils::mul_add_to(-(e->eta_), dEdW, e->V_);
        volume_utils::mul_add_to(-(e->wc_*e->eta_), e->W_, e->V_);
        volume_utils::add_to(e->V_, e->W_);

        // sparse filter
        if ( e->sparse_ != vec3i::one )
        {
            e->W_ = volume_utils::zero_out_nongrid(e->W_,e->sparse_);
        }

        // updated L2 norm
        e->norm_ = volume_utils::square_sum(e->W_);

        // FFT
        e->fft_.reset();
        if ( receives_fft_ )
        {
            e->fft_ = fftw::forward_pad(e->W_,e->in_->out_size());
        }
    }

    template<class Manager>
    void receive_f(double3d_ptr f, Manager task_manager)
    {
        {
            mutex::guard g(m_);
            ZI_ASSERT(in_received_<in_edges_.size());
            if ( in_received_ == 0 )
            {
                f_ = f;
                ++in_received_;
            }
            else
            {
                if ( f_ )
                {
                    double3d_ptr f2;
                    f_.swap(f2);
                    task_manager->insert(
                        zi::bind(
                            &node::template receive_f_sum<Manager>,
                            this, f, f2, task_manager),
                        forward_priority());
                }
                else
                {
                    f_ = f;
                    ++in_received_;
                }
            }

            if ( in_received_ == in_edges_.size() )
            {
                in_received_ = 0;
                ZI_ASSERT(f_);
                error_fn_->add_apply(bias_,f_);
                this->template run_forward<Manager>(task_manager);
            }
        }
    }

    template<class Manager>
    void receive_f_sum(double3d_ptr f1, double3d_ptr f2, Manager task_manager)
    {
        volume_utils::add_to(f1,f2);
        receive_f(f2, task_manager);
    }

    template<class Manager>
    void receive_f(complex3d_ptr f, Manager task_manager)
    {
        {
            mutex::guard g(m_);
            ZI_ASSERT(in_received_<in_edges_.size());
            if ( in_received_ == 0 )
            {
                fft_ = f;
                ++in_received_;
            }
            else
            {
                if ( fft_ )
                {
                    complex3d_ptr f2;
                    fft_.swap(f2);
                    task_manager->insert(
                        zi::bind(
                            &node::template receive_f_csum<Manager>,
                            this, f, f2, task_manager),
                        forward_priority());
                }
                else
                {
                    fft_ = f;
                    ++in_received_;
                }
            }

            if ( in_received_ == in_edges_.size() )
            {
                in_received_ = 0;
                ZI_ASSERT(fft_);
                vec3i s = in_edges_.front()->in_->out_size();

                double3d_ptr x = fftw::backward(std::move(fft_),s);

                // [TODO: zlateski]  normalize after cropping

                volume_utils::normalize(x);
                f_ = volume_utils::crop_right(x,size_);

                error_fn_->add_apply(bias_,f_);

                this->template run_forward<Manager>(task_manager);
            }
        }
    }

    template<class Manager>
    void receive_f_csum(complex3d_ptr f1, complex3d_ptr f2, Manager task_manager)
    {
        volume_utils::template add_to<complex3d_ptr>(f1,f2);
        receive_f(f2, task_manager);
    }

    template<class Manager>
    void receive_grad(double3d_ptr grad, Manager task_manager)
    {
        {
            mutex::guard g(m_);
            //std::cout << out_received_ << ' ' << out_edges_.size() << '\n';
            ZI_ASSERT(out_received_<out_edges_.size());
            if ( out_received_ == 0 )
            {
                dEdX_ = grad;
                ++out_received_;
            }
            else
            {
                if ( dEdX_ )
                {
                    double3d_ptr dEdX2;
                    dEdX_.swap(dEdX2);
                    task_manager->insert(
                        zi::bind(
                            &node::template receive_grad_sum<Manager>,
                            this, grad, dEdX2, task_manager),
                        backward_priority());
                }
                else
                {
                    dEdX_ = grad;
                    ++out_received_;
                }
            }

            if ( out_received_ == out_edges_.size() )
            {
                out_received_ = 0;
                ZI_ASSERT(dEdX_);
                this->template run_backward<Manager>(task_manager);
            }
        }
    }

    template<class Manager>
    void receive_grad_sum(double3d_ptr g1, double3d_ptr g2, Manager task_manager)
    {
        volume_utils::template add_to<double3d_ptr>(g1,g2);
        receive_grad(g2, task_manager);
    }

    template<class Manager>
    void receive_grad(complex3d_ptr grad, Manager task_manager)
    {
        {
            mutex::guard g(m_);

            ZI_ASSERT(out_received_<out_edges_.size());
            if ( out_received_ == 0 )
            {
                dEdX_fft_ = grad;
                ++out_received_;
            }
            else
            {
                if ( dEdX_fft_ )
                {
                    complex3d_ptr dEdX2;
                    dEdX_fft_.swap(dEdX2);
                    task_manager->insert(
                        zi::bind(
                            &node::template receive_grad_csum<Manager>,
                            this, grad, dEdX2, task_manager),
                        backward_priority());
                }
                else
                {
                    dEdX_fft_ = grad;
                    ++out_received_;
                }
            }

            if ( out_received_ == out_edges_.size() )
            {
                out_received_ = 0;
                ZI_ASSERT(dEdX_fft_);

                dEdX_ = volume_utils::normalize_flip(fftw::backward(std::move(dEdX_fft_),
                                                                        out_size()));

                this->template run_backward<Manager>(task_manager);
            }
        }
    }

    template<class Manager>
    void receive_grad_csum(complex3d_ptr g1, complex3d_ptr g2, Manager task_manager)
    {
        volume_utils::template add_to<complex3d_ptr>(g1,g2);
        receive_grad(g2, task_manager);
    }

    template<class Manager>
    std::size_t run_forward(Manager task_manager)
    {
        // do the filtering (currently hard-coded to use max-filtering)
        if ( filter_size_ != vec3i::one )
        {
            //ZI_ASSERT(filter_function_);
            std::pair<double3d_ptr, long3d_ptr> fandi =
                generic_filter(f_, filter_size_,
                               sparse_,
                               std::greater<double>());// filter_function_);
            f_ = fandi.first;
            filter_indices_ = fandi.second;
        }

        ZI_ASSERT((in_received_==0)&&(out_received_==0));
        ++pass_no_;

        if ( sends_fft_ && out_edges_.size() )
        {
            fft_ = fftw::forward(std::move(f_));
        }

        FOR_EACH( it, out_edges_ )
        {
            task_manager->insert(zi::bind(
                                        &node::template forward_edge<Manager>,
                                        this, *it, task_manager),
                                 (*it)->out_->forward_priority());
        }

        if ( waiters_ )
        {
            cv_.notify_all();
        }

        return pass_no_;
    }


    template<class Manager>
    std::size_t run_backward(Manager task_manager)
    {
        ZI_ASSERT((in_received_==0)&&(out_received_==0));
        ++pass_no_;

        dEdX_ = error_fn_->gradient(dEdX_, f_);

        if ( filter_size_ != vec3i::one )
        {
            ZI_ASSERT(filter_indices_);
            dEdX_ = do_filter_backprop(dEdX_,filter_indices_,real_filter_size_);
        }

        double dEdB = volume_utils::sum_all(dEdX_);

        // average gradients over the pixels in a minibatch
        double avg_dEdB = dEdB/szB_;

        // momentum & weight decay
        v_ = (mom_*v_) - (eta_*wc_*bias_) - (eta_*avg_dEdB);

        // bias update
        bias_ += v_;

        if ( receives_fft_ && in_edges_.size() )
        {
            vec3i s = in_edges_.front()->in_->out_size();
            volume_utils::flip(*dEdX_);
            dEdX_fft_ = fftw::forward_pad(dEdX_, s);
        }

        FOR_EACH(it, in_edges_)
        {
            task_manager->insert(zi::bind(
                                        &node::template backward_edge<Manager>,
                                        this, *it, task_manager),
                                 (*it)->in_->backward_priority());
        }

        if ( waiters_ )
        {
            cv_.notify_all();
        }
        return pass_no_;
    }

public:
    void add_out_edge(edge_ptr e)
    {
        out_edges_.push_back(e);
    }

    void add_in_edge(edge_ptr e)
    {
        in_edges_.push_back(e);
    }

    template<class Manager>
    std::size_t run_forward(double3d_ptr f, Manager task_manager)
    {
        mutex::guard g(m_);
        f_ = f;
        return this->template run_forward<Manager>(task_manager);
    }

    template<class Manager>
    std::size_t run_backward(double3d_ptr dEdX, Manager task_manager)
    {
        mutex::guard g(m_);
        dEdX_ = dEdX;
        return this->template run_backward<Manager>(task_manager);
    }

    double3d_ptr wait(std::size_t n)
    {
        mutex::guard g(m_);
        while ( pass_no_ < n )
        {
            ++waiters_;
            cv_.wait(m_);
            --waiters_;
        }

        if ( pass_no_ == n )
        {
            return f_;
        }
        else
        {
            throw std::logic_error("Pass somehow skipped!");
        }
    }

    vec3i size() const
    {
        return size_;
    }

    vec3i out_size() const
    {
        return size_ - real_filter_size_ + vec3i::one;
    }

    // for normalized initialization
    double fan_in() const
    {
        double ret = static_cast<double>(1);
        if ( count_in_edges() > 0 )
        {
            vec3i sz = in_edges_.front()->size();
            ret = static_cast<double>(sz[0]*sz[1]*sz[2]);
            ret *= count_in_edges();
        }
        return ret;
    }

    double fan_out() const
    {
        double ret = static_cast<double>(1);
        if ( count_out_edges() > 0 )
        {
            vec3i sz = out_edges_.front()->size();
            ret = static_cast<double>(sz[0]*sz[1]*sz[2]);
            ret *= count_out_edges();
        }
        return ret;
      }


// for exporting net
public:
    void print_bias( std::ostream& stream )
    {
        stream.write( reinterpret_cast<char*>(&bias_), sizeof(double) );
    }

    void print_eta( std::ostream& stream )
    {
        stream.write( reinterpret_cast<char*>(&eta_), sizeof(double) );
    }

    void print_v( std::ostream& stream )
    {
        stream.write( reinterpret_cast<char*>(&v_), sizeof(double) );
    }

    void print_out_edges( std::ostream& stream )
    {
        FOR_EACH( it, out_edges_ )
        {
            (*it)->print_W(stream); // W
        }
    }

    void print_out_nodes( std::ostream& stream )
    {
        out_edges_.front()->out_->print_eta(stream);    // eta
        FOR_EACH( it, out_edges_ )
        {
            (*it)->out_->print_bias(stream);            // bias
        }
    }

public:
    void sends_fft(bool b)
    {
        sends_fft_ = b;
        FOR_EACH( it, out_edges_ )
        {
            (*it)->out_->receives_fft_ = b;
        }
    }

    void receives_fft(bool b)
    {
        receives_fft_ = b;
        FOR_EACH( it, in_edges_ )
        {
            (*it)->in_->sends_fft_ = b;
        }
    }

    bool sends_fft() const
    {
        return sends_fft_;
    }

    bool receives_fft() const
    {
        return receives_fft_;
    }

    const vec3i& in_size() const
    {
        return size_;
    }

    void print_f() const
    {
        volume_utils::print(f_);
    }

}; // class node

}} // namespace zi::znn

#endif // ZNN_NODE_HPP_INCLUDED
