//
// Copyright (C) 2014  Kisuk Lee <kisuklee@mit.edu>
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

#ifndef ZNN_LEARNING_MONITOR_HPP_INCLUDED
#define ZNN_LEARNING_MONITOR_HPP_INCLUDED

#include "../../core/types.hpp"
#include "../../cost_fn/cost_fns.hpp"

namespace zi {
namespace znn {

class learning_monitor
{
private:
    typedef std::vector<double>         err_list;
    typedef std::vector<std::size_t>    iter_list;

private:
    std::string     name_;

    double          cost_sum_;
    double          CLSE_sum_;
    double          num_outs_;

    cost_fn_ptr     cost_fn_ ;
    err_list        cost_lst_;
    err_list        CLSE_lst_;  // CLSE: classification error    
    iter_list       iter_lst_;

public:
    double cost( std::list<double3d_ptr> outs,
                 std::list<double3d_ptr> lbls,
                 std::list<bool3d_ptr>   msks )
    {
        return cost_fn_->compute_cost(outs,lbls,msks) /
               cost_fn_->get_output_number(msks);
    }

    void update( std::list<double3d_ptr> outs,
                 std::list<double3d_ptr> lbls,
                 std::list<bool3d_ptr>   msks,
                 double cls_thresh = 0.5 )
    {
        cost_sum_ += cost_fn_->compute_cost(outs,lbls,msks);
        CLSE_sum_ += cost_fn_->compute_cls_error(outs,lbls,msks,cls_thresh);
        num_outs_ += cost_fn_->get_output_number(msks);
    }

    void reset()
    {
        cost_sum_ = static_cast<double>(0);
        CLSE_sum_ = static_cast<double>(0);
        num_outs_ = static_cast<double>(0);
    }

    void push_state( std::size_t iter )
    {        
        cost_lst_.push_back(cost_sum_/num_outs_);
        CLSE_lst_.push_back(CLSE_sum_/num_outs_);
        iter_lst_.push_back(iter);
    }    

    void report_state()
    {
        // cost
        cost_fn_->print_cost(cost_lst_.back());

        // classification error
        std::cout << "\tCLSE: " << CLSE_lst_.back() << std::endl;
    }

    void check( const std::string& path, std::size_t iter )
    {
        push_state(iter);
        report_state();
        save_state(path);
        reset();
    }

    void save_state( const std::string& path )
    {
        #define WRITE_FLAG  (std::ios::out | std::ios::binary)

        if ( iter_lst_.size() > 0 )
        {
            std::string sinfo = path + name_ + ".info";
            std::ofstream finfo(sinfo.c_str(), WRITE_FLAG);
            std::size_t iter = iter_lst_.size();
            finfo.write( reinterpret_cast<const char*>(&iter), sizeof(iter) );

            std::string siter = path + name_ + ".iter";
            std::ofstream fiter(siter.c_str(), WRITE_FLAG);
            fiter.write( reinterpret_cast<const char*>(&iter_lst_[0]),
                         sizeof(std::size_t)*iter_lst_.size() );

            std::string scost = path + name_ + ".err";
            std::ofstream fcost(scost.c_str(), WRITE_FLAG);
            fcost.write( reinterpret_cast<const char*>(&cost_lst_[0]),
                         sizeof(double)*cost_lst_.size() );

            std::string scls = path + name_ + ".cls";
            std::ofstream fcls(scls.c_str(), WRITE_FLAG);
            fcls.write( reinterpret_cast<const char*>(&CLSE_lst_[0]),
                        sizeof(double)*CLSE_lst_.size() );
        }
    }

    std::size_t load_state( const std::string& path, std::size_t index = 0 )
    {
        #define READ_FLAG  (std::ios::in | std::ios::binary)

        std::size_t iter = 0;

        std::string sinfo = path + name_ + ".info";
        std::ifstream finfo(sinfo.c_str(), READ_FLAG);
        if ( !finfo ) return 0;
        finfo.read( reinterpret_cast<char*>(&iter), sizeof(iter) );

        if ( iter > 0 )
        {
            if ( index > 0 ) iter = std::min(iter,index);

            std::string siter = path + name_ + ".iter";
            std::ifstream fiter(siter.c_str(), READ_FLAG);
            iter_lst_.resize(iter);
            fiter.read( reinterpret_cast<char*>(&iter_lst_[0]),
                        sizeof(std::size_t)*iter_lst_.size() );

            std::string scost = path + name_ + ".err";
            std::ifstream fcost(scost.c_str(), READ_FLAG);
            cost_lst_.resize(iter);
            fcost.read( reinterpret_cast<char*>(&cost_lst_[0]),
                        sizeof(double)*cost_lst_.size() );

            std::string scls = path + name_ + ".cls";
            std::ifstream fcls(scls.c_str(), READ_FLAG);
            CLSE_lst_.resize(iter);
            fcls.read( reinterpret_cast<char*>(&CLSE_lst_[0]),
                       sizeof(double)*CLSE_lst_.size() );

            // the lateset update number
            iter = iter_lst_.back();
        }

        return iter;
    }


public:
    void set_cost_fn( cost_fn_ptr cost_fn )
    {
        cost_fn_ = cost_fn;
    }


public:
    learning_monitor( const std::string& name,
                      cost_fn_ptr cost_fn = cost_fn_ptr(new square_cost_fn) )
        : name_(name)
        , cost_sum_(0)
        , CLSE_sum_(0)
        , num_outs_(0)
        , cost_fn_(cost_fn)
    {
    }

}; // class learning_monitor

typedef boost::shared_ptr<learning_monitor> learning_monitor_ptr;

}} // namespace zi::znn

#endif // ZNN_LEARNING_HPP_INCLUDED
