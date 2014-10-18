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

#ifndef ZNN_AFFINITY_GRAPH_HPP_INCLUDED
#define ZNN_AFFINITY_GRAPH_HPP_INCLUDED

#include "../../core/types.hpp"
#include "../../core/volume_pool.hpp"
#include "../../core/volume_utils.hpp"
#include "../../core/utils.hpp"

namespace zi {
namespace znn {

class affinity_graph;

typedef boost::shared_ptr<affinity_graph>  affinity_graph_ptr;
typedef std::list<double3d_ptr>            double3d_ptr_list ;

class affinity_graph
{
private:
    std::size_t         dim_  ;
    double3d_ptr_list   graph_;
    vec3i               size_ ;
    
    double  pos_affinity_edge_;
    double  neg_affinity_edge_;


// constructing affinity graph internally
private:
    // load affinity graph directly from file
    bool load( const std::string& fname )
    {
        std::string saffin = fname + ".affin";
        
        // size of the whole volume
        vec3i size_ = import_size_info(saffin);
        if ( size_ == vec3i::zero ) 
            return false;
        zi::wall_timer wt;
        std::cout << "Loading affinity graph from file..." << std::endl;
        std::cout << "Affinity graph size:\t" << size_ << std::endl;

        // load each affinity graph        
        for ( std::size_t i = 0; i < dim_; ++i )
        {
            std::ostringstream ssaffin;
            ssaffin << saffin << "." << i;

            double3d_ptr g = volume_pool.get_double3d(size_);
            volume_utils::load(g, ssaffin.str());
            graph_.push_back(g);
        }
        std::cout << "Completed. (Elapsed time: " 
                  << wt.elapsed<double>() << " secs)\n" << std::endl;
        return true;     
    }

    // load label from file ->
    // construct affinity graph from label
    bool load_from_label( const std::string& fname )
    {
        std::string slbl = fname + ".label";

        // size of the whole volume
        vec3i lblsz = import_size_info(fname);
        if ( lblsz == vec3i::zero )
            return false;
        std::cout << "Label size:\t" << lblsz << std::endl;
        size_ = lblsz - vec3i::one;

        // prepare volume for whole label
        double3d_ptr lbl = volume_pool.get_double3d(lblsz);

        // load each label
        if ( !volume_utils::load(lbl, slbl) )
            return false;

        // generate affinity graph from label
        construct_from_label(lbl);
        return true;
    }

    void construct_from_label( double3d_ptr lbl )
    {
        zi::wall_timer wt;
        std::cout << "Constructing affinity graph from label..." << std::endl;

        vec3i lblsz = size_of(lbl);
        size_ = lblsz - vec3i::one;
        std::cout << "Label size:\t" << lblsz << std::endl;
        std::cout << "Graph size:\t" << size_ << std::endl;

        for ( std::size_t i = 0; i < dim_; ++i )
        {
            double3d_ptr affin = volume_pool.get_double3d(size_);
            volume_utils::zero_out(affin);
            graph_.push_back(affin);
        }

        for ( std::size_t z = 0; z < size_[2]; ++z )
            for ( std::size_t y = 0; y < size_[1]; ++y )
                for ( std::size_t x = 0; x < size_[0]; ++x )
                {
                    std::list<double> v1_list;
                    v1_list.push_back((*lbl)[x][y+1][z+1]); // x-affinity
                    v1_list.push_back((*lbl)[x+1][y][z+1]); // y-affinity
                    v1_list.push_back((*lbl)[x+1][y+1][z]); // z-affinity
                    std::list<double>::iterator v1lit = v1_list.begin();

                    double v2 = (*lbl)[x+1][y+1][z+1];

                    FOR_EACH( git, graph_ )
                    {
                        double3d_ptr affin = *git;
                        double v1 = *v1lit;
                        if ( v1 == 0 && v2 == 0 )
                        {
                            (*affin)[x][y][z] = neg_affinity_edge_;
                        }
                        else
                        {
                            double diff = v1 - v2;
                            if ( diff == 0 )
                                (*affin)[x][y][z] = pos_affinity_edge_;
                            else
                                (*affin)[x][y][z] = neg_affinity_edge_;
                        }
                        ++v1lit;
                    }
                }

        std::cout << "Completed. (Elapsed time: " 
                  << wt.elapsed<double>() << " secs)\n" << std::endl;
    }


public:
    double3d_ptr_list get_labels()
    {
        return graph_;
    }

    affinity_graph_ptr get_subgraph( const vec3i& off, const vec3i& sz )
    {
        // extract sublabels
        double3d_ptr_list sublabels;
        FOR_EACH( it, graph_ )
        {
            sublabels.push_back(volume_utils::crop((*it), off, sz));
        }
        return affinity_graph_ptr(new affinity_graph(sublabels));
    }

    vec3i get_size() const
    {
        return size_;
    }


// Comparison
public:
    inline bool
    operator==( const affinity_graph& rhs )
    {
        if ( graph_ == rhs.graph_ )
            return true;

        double3d_ptr_list::const_iterator rit = rhs.graph_.begin();
        FOR_EACH( it, graph_ )
        {
            bool equal = (**it) == (**rit);
            if ( !equal )
                return false;
            ++rit;
        }

        return true;
    }


public:
    void save( const std::string& fname )
    {
        zi::wall_timer wt;
        std::cout << "<<<   affinity_graph::save   >>>" << std::endl;

        int i = 0;
        FOR_EACH( it, graph_ )
        {
            std::cout << "Now processing " << i << "th affinity..." << std::endl;
            std::ostringstream subname;
            subname << fname << ".affin." << i++;

            volume_utils::save((*it), subname.str());
        }

        std::string ssz = fname + ".affin";
        export_size_info( get_size(), ssz );

        std::cout << "Completed. (Elapsed time: " 
                  << wt.elapsed<double>() << " secs)\n" << std::endl;
    }


private:
    void set_affinity_edge_values( const double3d_ptr_list& graph )
    {
        pos_affinity_edge_ = static_cast<double>(1);
        neg_affinity_edge_ = static_cast<double>(0);

        FOR_EACH( it, graph )
        {
            std::size_t n = (*it)->num_elements();
            double min_val = *std::min_element((*it)->origin(), (*it)->origin() + n);
            double max_val = *std::max_element((*it)->origin(), (*it)->origin() + n);
            if( neg_affinity_edge_ > min_val )
                neg_affinity_edge_ = min_val;
            if( pos_affinity_edge_ < max_val )
                pos_affinity_edge_ = max_val;
        }
    }


public:
    affinity_graph( const std::string& fname, 
                    std::size_t dim = 3, 
                    double poslbl = static_cast<double>(1),
                    double neglbl = static_cast<double>(0) )
        : dim_(dim)
        , pos_affinity_edge_(poslbl)
        , neg_affinity_edge_(neglbl)
    {
        // if ( !load(fname) ) 
        // {
            STRONG_ASSERT( load_from_label(fname) );
        //     save(fname);
        // }
    }

    affinity_graph( double3d_ptr lbl, 
                    std::size_t dim = 3,
                    double poslbl = static_cast<double>(1),
                    double neglbl = static_cast<double>(0) )
        : dim_(dim)
        , pos_affinity_edge_(poslbl)
        , neg_affinity_edge_(neglbl)
    {
        construct_from_label(lbl);
    }

    affinity_graph( const double3d_ptr_list& graph )
    {
        ZI_ASSERT( graph.size() > 0 );

        graph_ = graph;
        dim_ = graph_.size();

        set_affinity_edge_values(graph);

        // each volume in the list is assumed to be the same in size
        size_ = size_of(graph_.front());
    }

}; // class affinity_graph

}} // namespace zi::znn

#endif // ZNN_AFFINITY_GRAPH_HPP_INCLUDED
