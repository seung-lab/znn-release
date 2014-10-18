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

#ifndef ZNN_NET_HPP_INCLUDED
#define ZNN_NET_HPP_INCLUDED

#include <boost/filesystem.hpp>

namespace zi {
namespace znn {

// forward declaration
class net;

typedef boost::shared_ptr<net> net_ptr;

class net
{
public:	
	// node & edge
	std::list<node_ptr> 		nodes_;
	std::list<edge_ptr> 		edges_;

	std::list<node_ptr> 		inputs_;
	std::list<node_ptr> 		outputs_;

	// node group & edge group
	std::list<node_group_ptr> 	node_groups_;
	std::list<edge_group_ptr> 	edge_groups_;

	std::list<node_group_ptr>	input_groups_;
	std::list<node_group_ptr>	output_groups_;

private:
	// initialization flag
	bool						initialized_;

	// network-wise parameters
	double						minibatch_size_;

	// path for re-load weight & bias
	std::string 	 			load_path_;


public:
	std::list<double3d_ptr> get_outputs( std::size_t n, bool softmax = false )
	{
		std::list<double3d_ptr> ret;
		FOR_EACH( it, output_groups_ )
		{
			std::list<double3d_ptr> output = (*it)->get_activations(n,softmax);
			ret.insert(ret.end(),output.begin(),output.end());
		}
		return ret;
	}


public:
	bool load()
	{
		return false;
	}

	void copy( net_ptr /* from */ )
	{

	}


public:
	void set_learning_rate( double eta )
	{
		// node groups
		FOR_EACH( it, node_groups_ )
		{
			(*it)->set_learning_rate(eta);
		}

		// edge groups
		FOR_EACH( it, edge_groups_ )
		{
			(*it)->set_learning_rate(eta);
		}
	}

	void set_momentum( double mom )
	{
		// node groups
		FOR_EACH( it, node_groups_ )
		{
			(*it)->set_momentum(mom);
		}

		// edge groups
		FOR_EACH( it, edge_groups_ )
		{
			(*it)->set_momentum(mom);
		}
	}

	void set_weight_decay( double wc )
	{
		// node groups
		FOR_EACH( it, node_groups_ )
		{
			(*it)->set_weight_decay(wc);
		}

		// edge groups
		FOR_EACH( it, edge_groups_ )
		{
			(*it)->set_weight_decay(wc);
		}
	}

	void set_minibatch_size( double size )
	{
		if ( minibatch_size_ != size )
		{
			minibatch_size_ = size;

			// node groups
			FOR_EACH( it, node_groups_ )
			{
				(*it)->set_minibatch_size(size);
			}

			// edge groups
			FOR_EACH( it, edge_groups_ )
			{
				(*it)->set_minibatch_size(size);
			}
		}
	}

	void set_load_path( const std::string& path )
	{
		load_path_ = path;
	}

	void force_fft( bool fft )
	{
		FOR_EACH( it, node_groups_ )
		{
			(*it)->receives_fft(fft);	
		}
	}


public:
	void add_node_group( node_group_ptr g )
	{
		node_groups_.push_back(g);
		FOR_EACH( it, g->nodes_ )
		{
			nodes_.push_back(*it);
		}
	}

	void add_edge_group( edge_group_ptr g )
	{
		edge_groups_.push_back(g);
		FOR_EACH( it, g->edges_ )
		{
			edges_.push_back(*it);
		}	
	}

	void find_inputs()
	{
		// input groups
		input_groups_.clear();
		FOR_EACH( it, node_groups_ )
		{
			if ( (*it)->count_in_connections() == 0 )
			{
				input_groups_.push_back(*it);
			}
		}

		// input nodes
		inputs_.clear();
		FOR_EACH( it, input_groups_ )
		{
			FOR_EACH( jt, (*it)->nodes_ )
			{
				if ( (*jt)->count_in_edges() == 0 )
				{
					inputs_.push_back(*jt);
				}
			}
		}
	}

	void find_outputs()
	{
		// output groups
		output_groups_.clear();
		FOR_EACH( it, node_groups_ )
		{
			if ( (*it)->count_out_connections() == 0 )
			{
				output_groups_.push_back(*it);
			}
		}

		// output nodes
		outputs_.clear();
		FOR_EACH( it, output_groups_ )
		{
			FOR_EACH( jt, (*it)->nodes_ )
			{
				if ( (*jt)->count_out_edges() == 0 )
				{
					outputs_.push_back(*jt);
				}
			}
		}
	}

	void check_dangling()
	{
		FOR_EACH( it, node_groups_ )
		{
			if ( (*it)->count_out_connections() == 0 &&
				 (*it)->count_in_connections() == 0 )
			{
				std::cout << "[net] check_dangling" << std::endl;
				std::string what = "Dangling node group [" + (*it)->name() + "]";
				throw std::invalid_argument(what);
			}
		}
	}


public:
	std::size_t count_inputs() const
	{
		return inputs_.size();
	}

	std::size_t count_outputs() const
	{
		return outputs_.size();
	}

	std::vector<vec3i> input_sizes() const
	{
		std::vector<vec3i> ret;
		
		if ( initialized() )
		{
			FOR_EACH( it, inputs_ )
			{
				ret.push_back((*it)->size());
			}
		}

		return ret;
	}

	std::vector<vec3i> output_sizes() const
	{
		std::vector<vec3i> ret;
		
		if ( initialized() )
		{
			FOR_EACH( it, outputs_ )
			{
				ret.push_back((*it)->out_size());
			}
		}

		return ret;
	}


public:
	bool initialized() const
	{
		return initialized_;
	}

	void disable_output_filtering()
	{
		FOR_EACH( it, output_groups_ )
		{
			(*it)->disable_filtering();
		}
	}

	void initialize( const vec3i& out_size = vec3i::one )
	{
		std::cout << "[net] initialize" << std::endl;

		// initialize weight & biases
		initialize_weight();

		// forward initialization
		FOR_EACH( it, input_groups_ )
		{
			(*it)->forward_sparse();
			(*it)->forward_layer_number();
		}

		// sort node groups by layer number
		// [02/10/2014 kisuklee] TODO
		// using lambda expression (C++11) to remove the need for sort_functor
		node_groups_.sort(sort_functor<node_group_ptr>());
		
		// backward initialization
		FOR_EACH( it, output_groups_ )
		{
			(*it)->backward_init(out_size);
		}
		
		// set fft optimization profile
		FOR_EACH( it, node_groups_ )
		{
			(*it)->set_fft_profile();
		}

		// display network architecture
		std::cout << '\n'; display();

		initialized_ = true;
	}

	void initialize_weight()
	{
		std::cout << "[net] intialize_weight" << std::endl;
		
		FOR_EACH( it, node_groups_ )
		{
			node_spec_ptr spec = (*it)->spec();
			if ( (*it)->is_loaded() )
			{
				// re-load bias
				(*it)->load_bias(load_path_);
			}
			else
			{
				node_factory.initialize_bias(*it);
			}	
		}

		FOR_EACH( it, edge_groups_ )
		{
			edge_spec_ptr spec = (*it)->spec();
			if ( (*it)->is_loaded() )
			{
				// re-load weight
				(*it)->load_weight(load_path_);
			}
			else
			{
				edge_factory.initialize_weight(*it);
			}
		}
	}

	void reload_weight( std::size_t idx )
	{
		std::cout << "[net] reload_weight (index = " << idx+1 << ")" << std::endl;
		
		FOR_EACH( it, node_groups_ )
		{
			node_spec_ptr spec = (*it)->spec();
			if ( (*it)->is_loaded() )
			{
				// re-load bias
				(*it)->load_bias(load_path_,idx);
			}
		}

		FOR_EACH( it, edge_groups_ )
		{
			edge_spec_ptr spec = (*it)->spec();
			if ( (*it)->is_loaded() )
			{
				// re-load weight
				(*it)->load_weight(load_path_,idx);
			}
		}
	}

	// crappy implementation
	// should be modified later
	void force_load()
	{
		std::cout << "[net] force_load" << std::endl;

		FOR_EACH( it, node_groups_ )
		{
			node_spec_ptr spec = (*it)->spec();			
			(*it)->load_bias(load_path_);
		}

		FOR_EACH( it, edge_groups_ )
		{
			edge_spec_ptr spec = (*it)->spec();			
			(*it)->load_weight(load_path_);
		}
	}

	void initialize_momentum()
	{
		std::cout << "[net] initialize_momentum" << std::endl;

		FOR_EACH( it, nodes_ )
		{
			(*it)->reset_v();
		}

		FOR_EACH( it, edges_ )
		{
			(*it)->reset_V();
		}
	}


public:
	void display() const
	{
		FOR_EACH( it, node_groups_ )
		{
			(*it)->display();
			std::cout << std::endl;

			FOR_EACH( jt, (*it)->out_ )
			{
				(*jt)->display();
				std::cout << std::endl;
			}
		}
	}


public:
	void save( std::string path, bool history = false )
	{
		if ( path == "" )
		{
			path = "./";
		}

		boost::filesystem::path save_dir(path);
		if ( !boost::filesystem::exists(save_dir) )
		{
    		boost::filesystem::create_directory(save_dir);
    	}

		STRONG_ASSERT(boost::filesystem::is_directory(save_dir));

		// nodes
		FOR_EACH( it, node_groups_ )
		{
			(*it)->save(path,history);
		}

		// edges
		FOR_EACH( it, edge_groups_ )
		{
			(*it)->save(path,history);
		}
	}


public:
	net()
		: nodes_()
		, edges_()
		, inputs_()
		, outputs_()
		, node_groups_()
		, edge_groups_()
		, input_groups_()
		, output_groups_()
		, initialized_(false)
		, minibatch_size_(static_cast<double>(1))
		, load_path_("")
	{}

}; // class net

}} // namespace zi::znn

#endif // ZNN_NET_HPP_INCLUDED
