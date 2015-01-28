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

#ifndef ZNN_OPTIONS_HPP_INCLUDED
#define ZNN_OPTIONS_HPP_INCLUDED

#include "../core/types.hpp"
#include "../cost_fn/cost_fns.hpp"
#include <zi/zargs/parser.hpp>

#include <boost/program_options.hpp>
#include <boost/tokenizer.hpp>
#include <boost/foreach.hpp>
#include <boost/filesystem.hpp>
#include <sstream>
#include <algorithm>


namespace zi {
namespace znn {

typedef std::vector<std::size_t> batch_list;

class network;

class options
{
private:
	std::string			train_range_str;
	std::string			test_range_str;

public:
	// [PATH]
	std::string			config_path;
	std::string			load_path;
	std::string 		data_path;
	std::string 		save_path;
	std::string 		hist_path;

	// [OPTIMIZE]
	std::size_t 		n_threads;
	bool				force_fft;
	bool				optimize_fft;

	// [TRAIN]
	batch_list			train_range;
	batch_list			test_range;
	vec3i 				outsz;
	std::string			dp_type;
	std::string			cost_fn;
	double				cost_fn_param;
	bool				data_aug;
	double 				cls_thresh;
	bool				softmax;
	bool				mirroring;

	// [UPDATE]
	double				force_eta;
	double				momentum;
	double				wc_factor;
	double				anneal_factor;
	std::size_t 		anneal_freq;
	bool				minibatch;
	bool				norm_grad;
	bool				rebalance;

	// [MONITOR]
	std::size_t 		n_iters;
	std::size_t 		check_freq;
	std::size_t 		test_freq;
	std::size_t 		test_samples;

	// [SCANNING]
	std::string			scanner;
	vec3i				scan_offset;
	vec3i				subvol_dim;	
	std::size_t 		weight_idx;
	bool				force_load;
	bool				out_filter;
	std::string 		outname;
	std::string 		subname;


private:
	// for parsing configuration file
	boost::program_options::options_description desc_;
	

public:	
	#define TEXT_WRITE 	(std::ios::out)
	#define TEXT_READ	(std::ios::in)

	void save() const
	{
		std::string fname = save_path + "options.config";
        std::ofstream fout(fname.c_str(),TEXT_WRITE);
        fout << (*this);
        fout.close();
	}

	bool build( const std::string& fpath )
	{
		std::ifstream fin(fpath.c_str(),TEXT_READ);
        if ( !fin )	return false;

        namespace po = boost::program_options;
        po::variables_map vm;
        po::store(po::parse_config_file(fin,desc_,true),vm);
        po::notify(vm);
        postprocess(vm);

        fin.close();
        return true;
	}


private:
	void initialize()
	{
		using namespace boost::program_options;

		desc_.add_options()
			// PATH
	        ("PATH.config",value<std::string>(&config_path)->default_value(""),"Network path")
	        ("PATH.load",value<std::string>(&load_path)->default_value(""),"Load path")
	        ("PATH.data",value<std::string>(&data_path)->default_value(""),"Data path")
	        ("PATH.save",value<std::string>(&save_path)->default_value(""),"Save path")
	        ("PATH.hist",value<std::string>(&hist_path)->default_value(""),"Save time-stamped network")
	        // OPTIMIZE
	        ("OPTIMIZE.n_threads",value<std::size_t>(&n_threads)->default_value(16),"Number of threads")
	        ("OPTIMIZE.force_fft",value<bool>(&force_fft)->default_value(false),"Force all FFTs")
	        ("OPTIMIZE.optimize_fft",value<bool>(&optimize_fft)->default_value(false),"FFT optimization")
	        // TRAIN
	        ("TRAIN.train_range",value<std::string>(&train_range_str)->default_value(""),"Train range")
	        ("TRAIN.test_range",value<std::string>(&test_range_str)->default_value(""),"Test range")
	        ("TRAIN.outsz",value<std::string>()->default_value("1,1,1"),"Output size")
	        ("TRAIN.dp_type",value<std::string>(&dp_type)->default_value("volume"),"Data provider")
	        ("TRAIN.cost_fn",value<std::string>(&cost_fn)->default_value("square"),"Cost function")
	        ("TRAIN.cost_fn_param",value<double>(&cost_fn_param)->default_value(0),"Cost function parameters")
	        ("TRAIN.data_aug",value<bool>(&data_aug)->default_value(false),"Data augmentation")
	        ("TRAIN.cls_thresh",value<double>(&cls_thresh)->default_value(0.5),"Classification threshold")
	        ("TRAIN.softmax",value<bool>(&softmax)->default_value(false),"Softmax")
	        ("TRAIN.mirroring",value<bool>(&mirroring)->default_value(false),"Boundary mirroring")
	        // UPDATE
	        ("UPDATE.force_eta",value<double>(&force_eta)->default_value(0),"Force the learning rate parameter")
	        ("UPDATE.momentum",value<double>(&momentum)->default_value(0),"Momentum")
	        ("UPDATE.wc",value<double>(&wc_factor)->default_value(0),"Weight decay")
	        ("UPDATE.anneal_factor",value<double>(&anneal_factor)->default_value(0),"Learning rate parameter annealing factor")
	        ("UPDATE.anneal_freq",value<std::size_t>(&anneal_freq)->default_value(0),"Learning rate parameter annealing frequency")
	        ("UPDATE.minibatch",value<bool>(&minibatch)->default_value(true),"Minibatch")
	        ("UPDATE.norm_grad",value<bool>(&norm_grad)->default_value(false),"Gradient normalization")
	        ("UPDATE.rebalance",value<bool>(&rebalance)->default_value(false),"Rebalancing")
	        // MONITOR
	        ("MONITOR.n_iters",value<std::size_t>(&n_iters)->default_value(1000000),"Number of training iterations")
	        ("MONITOR.check_freq",value<std::size_t>(&check_freq)->default_value(10),"Period for saving filter and display error")
	        ("MONITOR.test_freq",value<std::size_t>(&test_freq)->default_value(100),"Period for saving filter and display error")
	        ("MONITOR.test_samples",value<std::size_t>(&test_samples)->default_value(10),"Number of input patches for periodic testing")
	       	// SCAN
	       	("SCAN.scanner",value<std::string>(&scanner)->default_value("volume"),"Forward scanner")
	        ("SCAN.offset",value<std::string>()->default_value("0,0,0"),"Offset to start forward scanning")
	        ("SCAN.dim",value<std::string>()->default_value("0,0,0"),"Number of subvolumes for each dimension")
	        ("SCAN.weight_idx",value<std::size_t>(&weight_idx)->default_value(0),"Time-stamped network weight")	        
	        ("SCAN.force_load",value<bool>(&force_load)->default_value(true),"Force load")
	        ("SCAN.out_filter",value<bool>(&out_filter)->default_value(true),"Enable output filtering")
	        ("SCAN.outname",value<std::string>(&outname)->default_value("out"),"Output file name")
	        ("SCAN.subname",value<std::string>(&subname)->default_value(""),"Output file subname")
	        ;
	}

	void postprocess( boost::program_options::variables_map& vm )
	{
		std::cout << "\n[options] postprocess" << std::endl;

		zi::zargs_::parser<std::vector<std::size_t> > _parser;
		std::vector<std::size_t> target;
		std::string source;

		// path check
		path_check();

		// check validity of the number of threads
		if ( n_threads == 0 )
		{
			throw std::invalid_argument("Thread number should be greater than 0");
		}

		// train range
		source = vm["TRAIN.train_range"].as<std::string>();
		train_range = parse_batch_range(source);
		print_range("train_range",train_range);

		// test range
		source = vm["TRAIN.test_range"].as<std::string>();
		test_range = parse_batch_range(source);
		print_range("test_range ",test_range);

		// output size
		target.clear();		
		source = vm["TRAIN.outsz"].as<std::string>();
		if ( _parser.parse(&target,source) )
		{
			for ( std::size_t i = 0; i < target.size(); ++i )
			{
				outsz[i] = target[i];
			}
		}

		if ( outsz[0]*outsz[1]*outsz[2] == 0 )
		{
			std::string what = "Bad output size [" + vec3i_to_string(outsz) + "]";
			throw std::invalid_argument(what);
		}

		// scan offset
		target.clear();
		source = vm["SCAN.offset"].as<std::string>();
		if ( _parser.parse(&target,source) )
		{
			for ( std::size_t i = 0; i < target.size(); ++i )
			{
				scan_offset[i] = target[i];
			}
		}

		// subvolume dimension
		target.clear();
		source = vm["SCAN.dim"].as<std::string>();
		if ( _parser.parse(&target,source) )
		{
			for ( std::size_t i = 0; i < target.size(); ++i )
			{
				subvol_dim[i] = target[i];
			}
		}
	}


public:
	friend std::ostream&
	operator<<( std::ostream& os, const options& rhs )
	{
		return (os << "[PATH]\n"
        		   << "config=" << rhs.config_path << '\n'
        		   << "load=" << rhs.load_path << '\n'
        		   << "data=" << rhs.data_path << '\n'
        		   << "save=" << rhs.save_path << '\n'
        		   << "hist=" << rhs.hist_path << '\n'
        		   << "\n[OPTIMIZE]\n"
        		   << "n_threads=" << rhs.n_threads << '\n'
        		   << "force_fft=" << rhs.force_fft << '\n'
        		   << "optimize_fft=" << rhs.optimize_fft << '\n'
        		   << "\n[TRAIN]\n"
        		   << "train_range=" << rhs.train_range_str << '\n'
        		   << "test_range=" << rhs.test_range_str << '\n'
        		   << "outsz=" << vec3i_to_string(rhs.outsz) << '\n'
        		   << "dp_type=" << rhs.dp_type << '\n'
        		   << "cost_fn=" << rhs.cost_fn << '\n'
        		   << "cost_fn_param=" << rhs.cost_fn_param << '\n'
        		   << "data_aug=" << rhs.data_aug << '\n'
        		   << "cls_thresh=" << rhs.cls_thresh << '\n'
        		   << "softmax=" << rhs.softmax << '\n'
        		   << "mirroring=" << rhs.mirroring << '\n'
        		   << "\n[UPDATE]\n"
        		   << "force_eta=" << rhs.force_eta << '\n'
        		   << "momentum=" << rhs.momentum << '\n'
        		   << "wc=" << rhs.wc_factor << '\n'
        		   << "anneal_factor=" << rhs.anneal_factor << '\n'
        		   << "anneal_freq=" << rhs.anneal_freq << '\n'
        		   << "minibatch=" << rhs.minibatch << '\n'
        		   << "norm_grad=" << rhs.norm_grad << '\n'
        		   << "rebalance=" << rhs.rebalance << '\n'
        		   << "\n[MONITOR]\n"
        		   << "n_iters=" << rhs.n_iters << '\n'
        		   << "check_freq=" << rhs.check_freq << '\n'
        		   << "test_freq=" << rhs.test_freq << '\n'
        		   << "test_samples=" << rhs.test_samples << '\n'
        		   << "\n[SCAN]\n"
        		   << "scanner=" << rhs.scanner << '\n'
        		   << "offset=" << vec3i_to_string(rhs.scan_offset) << '\n'
        		   << "dim=" << vec3i_to_string(rhs.subvol_dim) << '\n'
        		   << "weight_idx=" << rhs.weight_idx << '\n'
        		   << "force_load=" << rhs.force_load << '\n'
        		   << "out_filter=" << rhs.out_filter << '\n'
        		   << "outname=" << rhs.outname << '\n'
        		   << "subname=" << rhs.subname << '\n');
	}


public:
	batch_list get_batch_range() const
	{
		batch_list ret = train_range;

		// set union over train and test range
		ret.insert(ret.end(),test_range.begin(),test_range.end());
		std::sort(ret.begin(),ret.end());
		ret.erase(std::unique(ret.begin(),ret.end()),ret.end());

		return ret;
	}

	// [01/28/2014 kisuklee]
	// The following create_something() methods should be replaced by
	// object factory design pattern later on
	cost_fn_ptr create_cost_function()
	{
		cost_fn_ptr ret;	
		if ( cost_fn == "square" )
		{
			ret = cost_fn_ptr(new square_cost_fn);
		}
	    else if ( cost_fn == "cross_entropy" )
	    {
	        ret = cost_fn_ptr(new cross_entropy_cost_fn);
	    }
	    else if ( cost_fn == "binomial_cross_entropy" )
	    {
	        ret = cost_fn_ptr(new binomial_cross_entropy_cost_fn);
	    }
	    else if ( cost_fn == "square_square" )
	    {
	    	ret = cost_fn_ptr(new square_square_cost_fn(cost_fn_param));
	    }
	    else
	    {
	    	std::string what = "Unknown cost function [" + cost_fn + "]";
			throw std::invalid_argument(what);
	    }
	    return ret;
	}


private:
	batch_list parse_batch_range( const std::string s )
	{
		batch_list ret;

		using namespace boost;
		char_separator<char> sep(",");
		tokenizer< char_separator<char> > tokens(s, sep);
		BOOST_FOREACH( const std::string& t, tokens )
		{
			// parsing
			std::string::size_type pos = t.find("-");
			if ( pos != std::string::npos )
			{					
				batch_list lst = parse_range_str(t);
				ret.insert(ret.end(), lst.begin(), lst.end());
			}
			else
			{
				std::size_t n;
				std::istringstream convert(t);
				if ( !(convert >> n) )
				{
					std::string what = "Invalid range [" + t + "]";
					throw std::invalid_argument(what);
				}
				ret.push_back(n);
			}
		}

		// unique and sorted
		std::sort(ret.begin(), ret.end());
		ret.erase(std::unique(ret.begin(), ret.end()), ret.end());
		return ret;
	}

	batch_list parse_range_str( const std::string s )
	{	
		batch_list ret;

		std::string::size_type pos = s.find("-");
		ZI_ASSERT( pos != std::string::npos );

		std::string s1 = s.substr(0,pos);
		std::string s2 = s.substr(pos+1);

		std::size_t n1, n2;
		std::istringstream convert1(s1);
		std::istringstream convert2(s2);
		if ( !(convert1 >> n1) || !(convert2 >> n2) )
		{
			std::string what = "Invalid range [" + s + "]";
			throw std::invalid_argument(what);
		}
		
		for ( std::size_t i = n1; i <= n2; ++i )
			ret.push_back(i);

		return ret;
	}

	void print_range( const std::string& name, const batch_list& range )
	{
		std::cout << name << " [";
		if ( range.empty() )
		{
			std::cout << "empty]" << std::endl;
			return;
		}

		std::cout << range.at(0);
		for ( std::size_t i = 1; i < range.size(); ++i )
		{
			std::cout << "," << range.at(i);
		}
		std::cout << "]" << std::endl;
	}


public:
	void path_check()
	{
		std::cout << "\n[options] path_check" << std::endl;
		check_config_path();
		check_load_path();
		// data path will be checked later
		check_save_path();
		check_hist_path();
	}


private:

	void check_config_path() const
	{
		boost::filesystem::path config_file(config_path);

		// config path is not allowed to be empty
		if ( !boost::filesystem::exists(config_file) )
		{
			std::string what = "Non-existent config path [" + config_path + "]";
			throw std::invalid_argument(what);
		}		
		
		if ( boost::filesystem::is_directory(config_file) )
		{
			std::string what = "Non-file config path [" + config_path + "]";
			throw std::invalid_argument(what);
		}

		std::cout << "Config path [" << config_path << "]" << std::endl;
	}

	void check_load_path()
	{
		// load path is allowed to be empty
		if ( load_path.empty() )
		{
			std::cout << "Load path   [empty]" << std::endl;
			return;
		}

		boost::filesystem::path load_dir(load_path);

		if ( !boost::filesystem::exists(load_dir) )
		{
			std::string what = "Non-existent load path [" + load_path + "]";
			throw std::invalid_argument(what);
		}
		
		if ( !boost::filesystem::is_directory(load_dir) )
		{
			std::string what = "Non-directory load path [" + load_path + "]";
			throw std::invalid_argument(what);
		}
		
		if ( *load_path.rbegin() != '/' )
		{
			load_path = load_path + "/";
		}
		std::cout << "Load path   [" << load_path << "]" << std::endl;
	}

	void check_save_path()
	{
		// save path is allowed to be empty
		if ( save_path.empty() )
		{
			std::cout << "Save path   [empty]" << std::endl;
			return;
		}

		boost::filesystem::path save_dir(save_path);

		// save path is not allowed to be empty		
		if ( !boost::filesystem::exists(save_dir) )
		{
			std::string what = "Non-existent save path [" + save_path + "]";
			throw std::invalid_argument(what);
		}
		
		if ( !boost::filesystem::is_directory(save_dir) )
		{
			std::string what = "Non-directory save path [" + save_path + "]";
			throw std::invalid_argument(what);
		}

		if ( *save_path.rbegin() != '/' )
		{
			save_path = save_path + "/";
		}
		std::cout << "Save path   [" << save_path << "]" << std::endl;
	}

	void check_hist_path()
	{
		// hist path is allowed to be empty
		if ( hist_path.empty() )
		{
			std::cout << "Hist path   [empty]" << std::endl;
			return;
		}
		
		boost::filesystem::path hist_dir(hist_path);

		if ( !boost::filesystem::exists(hist_dir) )
		{
			std::string what = "Non-existent hist path [" + hist_path + "]";
			throw std::invalid_argument(what);
		}
		
		if ( !boost::filesystem::is_directory(hist_dir) )
		{
			std::string what = "Non-directory hist path [" + hist_path + "]";
			throw std::invalid_argument(what);
		}
		
		if ( *hist_path.rbegin() != '/' )
		{
			hist_path = hist_path + "/";
		}
		std::cout << "Hist path   [" << hist_path << "]" << std::endl;
	}


public:
	options( const std::string& path )
		: outsz(vec3i::one)
		, scan_offset(vec3i::zero)
		, subvol_dim(vec3i::zero)
	{
		initialize();
		
		if ( !build(path) )
		{
			std::string what 
				= "Failed to build training options from the file [" + path + "]";
			throw std::invalid_argument(what);
		}
	}

friend class network;

}; // class options

typedef boost::shared_ptr<options> options_ptr;

}} // namespace zi::znn

#endif // ZNN_OPTIONS_HPP_INCLUDED