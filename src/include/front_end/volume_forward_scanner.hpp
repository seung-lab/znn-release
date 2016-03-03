//
// Copyright (C)      2016  Kisuk Lee           <kisuklee@mit.edu>
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

#include "forward_scanner.hpp"
#include "volume_data.hpp"
#include "../network/parallel/network.hpp"

namespace znn { namespace v4 {

template <typename T>
class volume_forward_scanner
{
private:
    typedef std::vector<volume_data<T>>         input_type      ;
    typedef std::map<size_t,rw_volume_data<T>>  output_type     ;
    typedef std::pair<vec3i,size_t>             layer_size_type ;
    typedef std::pair<vec3i,std::set<size_t>>   layer_spec_type ;
    typedef std::vector<cube_p<T>>              tensor_type     ;

private:
    std::map<std::string, layer_size_type>      inputs_size_ ;
    std::map<std::string, layer_size_type>      outputs_size_;

    std::map<std::string, layer_spec_type>      layers_spec_;
    std::vector<std::string>                    layers_name_;

    std::map<std::string, input_type>           inputs_ ;
    std::map<std::string, output_type>          outputs_;

    // scan parameters
    vec3i scan_offset_;
    vec3i scan_stride_;
    vec3i scan_grid_  ;

    std::vector<std::set<size_t>> scan_coords_;

    box range_;

    parallel_network::network * net_;


public:
    void scan()
    {
        for ( auto z: scan_coords_[0] )
            for ( auto y: scan_coords_[1] )
                for ( auto x: scan_coords_[2] )
                    scan(vec3i(z,y,x));
    }

private:
    void scan( vec3i const & loc )
    {
        zi::wall_timer wt;
        std::cout << "Scanning [" << loc - min_corner() << "] ... ";

        wt.reset();
        {
            net_->forward(pull(loc));
            push(loc, net_->get_featuremaps(layers_name_));
        }
        auto elapsed = wt.elapsed<double>();

        std::cout << "done. (elapsed: " << elapsed << ")\n";
    }

    // pull inputs from the specified location
    std::map<std::string, tensor_type> pull( vec3i const & loc )
    {
        std::map<std::string, tensor_type> ret;

        for ( auto& i: inputs_size_ )
        {
            auto& name = i.first;
            auto  size = i.second.second;

            for ( size_t i = 0; i < size; ++i )
            {
                ret[name][i] = inputs_[name][i].get_patch(loc);
            }
        }

        return ret;
    }

    // push outputs to the specified location
    void push( vec3i const & loc, std::map<std::string, tensor_type> & data )
    {
        for ( auto& l: layers_spec_ )
        {
            auto& name  = l.first;
            auto& range = l.second.second;

            for ( auto& v: range )
            {
                ZI_ASSERT(outputs_[name][v].count()!=0);
                outputs_[name][v].set_patch(loc, data[name][v-1]);
            }
        }
    }


public:
    std::map<std::string, tensor_type> outputs( bool auto_crop = true )
    {
        // TODO(lee): return auto-crop results
    }

    std::map<std::string, layer_spec_type> layer_spec()
    {
        return layers_spec_;
    }


public:
    void add_input( std::string const & name,
                    tensor_type const & data,
                    vec3i const & offset = vec3i::zero )
    {
        ZI_ASSERT(inputs_size_.count(name)!=0);
        ZI_ASSERT(inputs_.count(name)==0);

        auto patch_size = inputs_size_[name].first;

        auto& input = inputs_[name];
        for ( auto& d: data )
        {
            input.emplace_back(d, patch_size, offset);
        }

        ZI_ASSERT(input.size()!=0);
        ZI_ASSERT(input.size()==inputs_size_[name].second);

        // update valid range
        box const & range = input[0].range();
        range_ = range_.empty() ? range : range_.intersect(range);
    }

    // setup after adding inputs
    void setup()
    {
        setup_scan_stride();

        setup_scan_coords(0); // z-coordinate
        setup_scan_coords(1); // y-coordinate
        setup_scan_coords(2); // x-coordinate

        prepare_outputs();
    }

private:
    void setup_scan_stride()
    {
        vec3i vmin = min_corner();

        box a;
        for ( auto& o: outputs_size_ )
        {
            vec3i outsz = o.second.first;
            box b = box::centered_box(vmin, outsz);

            // update output box
            a = a.empty() ? b : a.intersect(b);
        }

        scan_stride_ = a.size();
    }

    void setup_scan_coords( size_t dim )
    {
        // 0: z-dimension
        // 1: y-dimension
        // 2: x-dimension
        ZI_ASSERT(dim < 3);

        // min & max corners of the scan range
        vec3i vmin = range_.min() + scan_offset_;
        vec3i vmax = range_.max();

        // min & max coordinates
        auto cmin = vmin[dim];
        auto cmax = vmax[dim];

        // dimension-specific parameters
        auto& offset = scan_offset_[dim];
        auto& stride = scan_stride_[dim];
        auto& grid   = scan_grid_[dim];
        auto& coords = scan_coords_[dim];

        // stride should be minimum output dimension
        // (assuming outputs with different sizes)
        std::set<size_t> out_dims;
        for ( auto& o: outputs_size_ )
            out_dims.insert(o.second.first[dim]);
        stride = *out_dims.begin();

        // automatic full spanning
        if ( !grid )
        {
            grid = (cmax - cmin)/stride;
            coords.insert(cmax-1); // offcut
        }

        // scan coordinates
        size_t coord = cmin;
        for ( size_t i = 0; i < grid; ++i  )
        {
            coords.insert(coord);
            coord += stride;
        }

        // sanity check
        ZI_ASSERT((cmin + (grid-1)*stride) < cmax);
    }

    void prepare_outputs()
    {
        for ( auto& l: layers_spec_ )
        {
            vec3i outsz = l.second.first;
            box a = box::centered_box(min_corner(), outsz);
            box b = box::centered_box(max_corner(), outsz);
            add_output(l.first, a+b);
        }
    }

    void add_output( std::string const & name, box const & bbox )
    {
        ZI_ASSERT(layers_spec_.count(name)!=0);
        ZI_ASSERT(outputs_.count(name)==0);

        auto patch_size = layers_spec_[name].first;
        auto range      = layers_spec_[name].second;

        auto& output = outputs_[name];
        for ( auto& v: range )
        {
            auto data = get_cube<T>(bbox.size());
            fill(*data,0);
            output.emplace(v, rw_volume_data<T>(data, patch_size, bbox.min()));
        }

        ZI_ASSERT(output.size()!=0);
        ZI_ASSERT(output.size()==range.size());
    }

    vec3i min_corner() const
    {
        vec3i r;

        r[0] = *scan_coords_[0].begin();
        r[1] = *scan_coords_[1].begin();
        r[2] = *scan_coords_[2].begin();

        return r;
    }

    vec3i max_corner() const
    {
        vec3i r;

        r[0] = *scan_coords_[0].rbegin();
        r[1] = *scan_coords_[1].rbegin();
        r[2] = *scan_coords_[2].rbegin();

        return r;
    }


private:
    void parse_spec( std::string const & fname )
    {
        auto layers_size = net_->layers();

        std::vector<options> nodes, edges;

        if ( fname.empty() )
        {
            for ( auto& o: outputs_size_ )
            {
                options op;
                op.push("name",o.first);
                nodes.push_back(op);
            }
        }
        else
        {
            parse_net_file(nodes, edges, fname);
        }

        for ( auto& n: nodes )
        {
            auto& name = n.require_as<std::string>("name");
            auto range = n.optional_as<std::string>("range","");

            layers_name_.push_back(name);
            layers_spec_[name].first = layers_size[name].first;
            if ( range.empty() )
            {
                size_t n = layers_size[name].second;
                for ( size_t i = 1; i <= n; ++i )
                    layers_spec_[name].second.insert(i);
            }
            else
            {
                layers_spec_[name].second =
                    parse_number_set<size_t>(range);
            }
        }
    }


public:
    explicit
    volume_forward_scanner( parallel_network::network * net,
                            vec3i const & offset = vec3i::zero,
                            vec3i const & grid = vec3i::zero,
                            std::string const & fname = "" )
        : net_(net)
        , inputs_size_(net->inputs())
        , outputs_size_(net->outputs())
        , scan_offset_(offset)
        , scan_grid_(grid)
    {
        parse_spec(fname);
    }

    ~volume_forward_scanner() {}

}; // class volume_forward_scanner

}} // namespace znn::v4