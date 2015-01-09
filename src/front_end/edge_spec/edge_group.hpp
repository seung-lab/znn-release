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

#ifndef ZNN_EDGE_GROUP_HPP_INCLUDED
#define ZNN_EDGE_GROUP_HPP_INCLUDED

#include "../../core/node.hpp"
#include "edge_spec.hpp"


namespace zi {
namespace znn {

// forward declaration
class node_group;

typedef boost::shared_ptr<node_group> node_group_ptr;

class edge_group
{
private:	
	const std::string		name_;
	edge_spec_ptr			spec_;
	std::vector<edge_ptr>	edges_;

	node_group_ptr			source_;
	node_group_ptr			target_;

	vec3i					sparse_;

	// weight loaded
	bool					loaded;


public:
	void set_learning_rate( double eta )
	{
		spec_->eta = eta;
		FOR_EACH( it, edges_ )
		{
			(*it)->set_eta(eta);
		}
	}

	void set_momentum( double mom )
	{
		spec_->mom = mom;
		FOR_EACH( it, edges_ )
		{
			(*it)->set_momentum(mom);
		}
	}

	void set_weight_decay( double wc )
	{
		spec_->wc = wc;
		FOR_EACH( it, edges_ )
		{
			(*it)->set_weight_decay(wc);
		}
	}

	void set_minibatch_size( double sz )
	{
		FOR_EACH( it, edges_ )
		{
			(*it)->set_minibatch_size(sz);
		}
	}


// only allow node_group/edge_factory to modify edge_group
// by declaring node_group/edge_factory as friends
private:
	void set_spec( edge_spec_ptr spec )
	{
		spec_ = spec;
	}

	void add_edge( edge_ptr edge )
	{
		edges_.push_back(edge);
	}

	void set_sparse( const vec3i& s )
	{
		if ( s != sparse_ )
		{
			sparse_ = s;
			FOR_EACH( it, edges_ )
			{
				(*it)->set_sparse(sparse_);
			}
		}
	}


public:
	std::size_t count() const
	{
		return edges_.size();
	}

	std::string name() const
	{
		return name_;
	}

	edge_spec_ptr spec()
	{
		return spec_;
	}

	vec3i filter_size() const
	{
		return spec_->size;
	}

	vec3i real_filter_size() const
	{
		return spec_->real_filter_size(sparse_);
	}

	bool is_loaded() const
	{
		return loaded;
	}


public:
	void save( const std::string& path, bool history = false ) const
	{
		spec_->save(path);	// save edge specification
		save_weight(path);	// save weight

		if ( history )
		{
			accumulate_weight(path);
		}
	}

private:
	#define BINARY_WRITE (std::ios::out | std::ios::binary)
	#define BINARY_READ	 (std::ios::in | std::ios::binary)
	#define BINARY_ACCUM (BINARY_WRITE | std::ios::app)

	void save_weight( const std::string& path ) const
	{
		std::string fpath = path + name_ + ".weight";
        std::ofstream fout(fpath.c_str(), BINARY_WRITE);

        FOR_EACH( it, edges_ )
        {
        	(*it)->print_W(fout);
        }
        fout.close();
	}

	void accumulate_weight( const std::string& path ) const
	{
		std::string fpath = path + name_ + ".weight.hist";
        std::ofstream fout(fpath.c_str(), BINARY_ACCUM);

        FOR_EACH( it, edges_ )
        {
        	(*it)->print_W(fout);
        }
        fout.close();
	}

	bool load_spec( const std::string& path )
	{
		std::string fpath = path + name_ + ".spec";
		bool ret = spec_->build(fpath);
		return ret;
	}

	bool load_weight( std::ifstream& fin )
	{
		STRONG_ASSERT( fin );

		FOR_EACH( it, edges_ )
		{
			vec3i ws = (*it)->size();
			double3d_ptr W = volume_pool.get_double3d(ws);
			double d;
			for ( std::size_t z = 0; z < ws[2]; ++z )
				for ( std::size_t y = 0; y < ws[1]; ++y )
					for ( std::size_t x = 0; x < ws[0]; ++x )
                    {
                    	fin.read(reinterpret_cast<char*>(&d),sizeof(d));
                        (*W)[x][y][z] = d;
                    }

            // vec3i s = (*it)->get_sparse();
			// (*it)->set_W(W);
			// (*it)->set_sparse(s);
            (*it)->reset_W(W);
		}

		fin.close();
		loaded = true;
		return true;
	}

	bool load_weight( const std::string& path )
	{
		std::string fpath = path + name_ + ".weight";
		std::ifstream fin(fpath.c_str(), BINARY_READ);
		if ( !fin ) return false;

		return load_weight(fin);
	}

	bool load_weight( const std::string& path, std::size_t idx )
	{
		std::string fpath = path + name_ + ".weight.hist";
		std::ifstream fin(fpath.c_str(), BINARY_READ);
		if ( !fin ) return false;

		// get length of the file
		fin.seekg(0,fin.end);
		int64_t len = fin.tellg();
		STRONG_ASSERT( len > 0 );
		
		// idx validity check
		vec3i ws = edges_.front()->size();
		std::size_t sz = sizeof(double)*(ws[0]*ws[1]*ws[2])*count();
		STRONG_ASSERT( len % sz == 0 );
		STRONG_ASSERT( len/sz > idx );

		// seek
		int64_t pos = sz*idx;
		fin.seekg(pos,fin.beg);

		return load_weight(fin);
	}


public:
	void display() const
	{
		std::cout << "[" << name() << "]" << std::endl;
		
		// Kernel
	  	std::cout << "Kernel size:\t\t" << spec_->size << " x "
	  			  << count() << std::endl;

	  	vec3i sparse = edges_.front()->get_sparse();
	  	if ( sparse != vec3i::one )
	  	{
	  		std::cout << "Sparseness:\t\t" << sparse << std::endl;
		  	std::cout << "Real filter size:\t" 
		  			  << spec_->real_filter_size(sparse) << std::endl;
		}
	}


// only allow edge factory to create edge_group
// by declaring edge factory as a friend
private:
	edge_group( const std::string& name,
				node_group_ptr source,
				node_group_ptr target )
		: name_(name)
		, spec_()
		, edges_()
		, source_(source)
		, target_(target)
		, sparse_(vec3i::one)
		, loaded(false)
		// , initializer_()
	{
		set_spec(edge_spec_ptr(new edge_spec(name)));
		// std::cout << "edge_group " << name_ << " has created!" << std::endl;
	}

	friend class edge_factory_impl;
	friend class node_group;
	friend class net;

}; // class edge_group

typedef boost::shared_ptr<edge_group> edge_group_ptr;

}} // namespace zi::znn

#endif // ZNN_EDGE_GROUP_HPP_INCLUDED
