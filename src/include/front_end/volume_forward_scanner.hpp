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

#include "volume_data.hpp"
#include "volume_dataset.hpp"
#include "../network/parallel/network.hpp"

namespace znn { namespace v4 {

template <typename T>
class volume_forward_scanner
{
private:
    typedef std::unique_ptr<rw_volume_data<T>> rw_volume_p;

private:
    struct output_spec
    {
        vec3i               dim;
        std::set<size_t>    range;
    };

private:
    parallel_network::network *         net_    ;
    std::shared_ptr<volume_dataset<T>>  inputs_ ;
    std::map<std::string, rw_volume_p>  outputs_;
    std::map<std::string, output_spec>  spec_   ;
    std::vector<std::string>            keys_   ;

    // scan parameters
    vec3i                               scan_offset_;
    vec3i                               scan_stride_;
    vec3i                               scan_grid_  ;
    std::vector<std::set<size_t>>       scan_coords_;


public:
    sample<T> scan()
    {
        zi::wall_timer wt;

        size_t i = 1;
        size_t n = scan_coords_[0].size()*
                   scan_coords_[1].size()*
                   scan_coords_[2].size();

        for ( auto z: scan_coords_[0] )
            for ( auto y: scan_coords_[1] )
                for ( auto x: scan_coords_[2] )
                {
                    std::cout << "Scanning (" << i++ << "/" << n << ") ...";
                    wt.reset();
                    {
                        scan(vec3i(z,y,x));
                    }
                    auto elapsed = wt.elapsed<double>();
                    std::cout << "done. (elapsed: " << elapsed << ")\n";
                }

        return outputs();
    }

private:
    void scan( vec3i const & loc )
    {
        auto ins = inputs_->get_sample(loc);
        net_->forward(std::move(ins));
        push(loc, net_->get_featuremaps(keys_));
    }

    // push outputs to the specified location
    void push( vec3i const & loc, sample<T> data )
    {
        for ( auto& s: spec_ )
        {
            auto& name  = s.first;
            auto& range = s.second.range;

            ZI_ASSERT(data.count(name)!=0);

            for ( auto& i: range )
            {
                std::string fmap = name + ":" + std::to_string(i);
                ZI_ASSERT(outputs_.count(fmap)!=0);
                outputs_[fmap]->set_patch(loc, data[name][i-1]);
            }
        }
    }

    sample<T> outputs( bool auto_crop = true )
    {
        box bcrop;
        if ( auto_crop )
        {
            for ( auto& o: outputs_ )
            {
                auto b = o.second->bbox();
                bcrop = bcrop.empty() ? b : bcrop.intersect(b);
            }
        }

        sample<T> ret;
        for ( auto& o: outputs_ )
        {
            auto vol = o.second->get_data();
            if ( auto_crop )
                vol = crop(*vol, bcrop.min(), bcrop.size());
            tensor<T> t  = {vol};
            ret[o.first] = t;
        }
        return ret;
    }


private:
    void setup()
    {
        inputs_->set_spec(net_->inputs());

        // order is important!
        setup_scan_stride();
        setup_scan_coords();
        prepare_outputs();
    }

    // scan stride should be the size of intersection of all outputs
    void setup_scan_stride()
    {
        box a; // empty box
        for ( auto& s: spec_ )
        {
            box b = box::centered_box(vec3i::zero, s.second.dim);
            a = a.empty() ? b : a.intersect(b);
        }
        scan_stride_ = a.size();
    }

    void setup_scan_coords()
    {
        setup_scan_coords(0); // z-coordinate
        setup_scan_coords(1); // y-coordinate
        setup_scan_coords(2); // x-coordinate
    }

    void setup_scan_coords( size_t dim )
    {
        // 0: z-dimension
        // 1: y-dimension
        // 2: x-dimension
        ZI_ASSERT(dim < 3);

        // min & max corners of the scan range
        vec3i vmin = inputs_->range().min() + scan_offset_;
        vec3i vmax = inputs_->range().max();

        // min & max coordinates
        auto cmin = vmin[dim];
        auto cmax = vmax[dim];

        // dimension-specific parameters
        auto & stride = scan_stride_[dim];
        auto & grid   = scan_grid_[dim];
        auto & coords = scan_coords_[dim];

        // automatic full spanning
        if ( !grid )
        {
            grid = (cmax - cmin - 1)/stride + 1;
            coords.insert(cmax-1); // offcut
        }

        // scan coordinates
        for ( size_t i = 0; i < grid; ++i  )
        {
            size_t coord = cmin + i*stride;
            if ( coord >= cmax ) break;
            coords.insert(coord);
        }

        // sanity check
        ZI_ASSERT((cmin + (grid-1)*stride) < cmax);
    }

    void prepare_outputs()
    {
        vec3i rmin;
        rmin[0] = *(scan_coords_[0].begin());
        rmin[1] = *(scan_coords_[1].begin());
        rmin[2] = *(scan_coords_[2].begin());

        vec3i rmax;
        rmax[0] = *(scan_coords_[0].rbegin());
        rmax[1] = *(scan_coords_[1].rbegin());
        rmax[2] = *(scan_coords_[2].rbegin());

        for ( auto& l: spec_ )
        {
            auto const & name = l.first;
            auto const & dim  = l.second.dim;

            box a = box::centered_box(rmin, dim);
            box b = box::centered_box(rmax, dim);

            add_output(name, a+b);
        }
    }

    void add_output( std::string const & name, box const & bbox )
    {
        ZI_ASSERT(spec_.count(name)!=0);

        auto const & dim   = spec_[name].dim;
        auto const & range = spec_[name].range;

        for ( auto& i: range )
        {
            auto data = get_cube<T>(bbox.size());
            fill(*data,0);

            // assign unique name to each feature map
            //  "layer_name:feature_map_number"
            //  e.g. nconv1:1
            std::string fmap = name + ":" + std::to_string(i);
            outputs_[fmap] =
                std::make_unique<rw_volume_data<T>>(data, dim, bbox.min());
        }
    }


private:
    void parse_spec( std::string const & fname )
    {
        auto layers = net_->layers();
        auto inputs = net_->inputs();

        std::vector<options> nodes, edges;

        parse_net_file(nodes, edges, fname);

        for ( auto& n: nodes )
        {
            auto name  = n.require_as<std::string>("name");
            auto type  = n.optional_as<std::string>("type","");
            auto range = n.optional_as<std::string>("range","");

            // skip non-existing layers
            if ( layers.count(name) == 0 ) continue;

            // skip input layers
            if ( inputs.count(name) != 0 ) continue;

            // parse range
            auto n = layers[name].second;
            if ( range.empty() )
            {
                // default range: 1-n (all nodes)
                range = "1-" + std::to_string(n);
            }

            auto range = parse_number_set<size_t>(range);

            // remove out-of-range indices
            for ( auto it = range.begin(); it != range.end(); )
            {
                if ( *it > n )
                    it = range.erase(it);
                else
                    ++it;
            }

            spec_[name].dim   = layers[name].first;
            spec_[name].range = range;
            keys_.push_back(name);
        }
    }


public:
    explicit
    volume_forward_scanner( parallel_network::network * net,
                            std::shared_ptr<volume_dataset<T>> dataset,
                            std::string const & fname,
                            vec3i const & offset = vec3i::zero,
                            vec3i const & grid   = vec3i::zero )
        : net_(net)
        , inputs_(dataset)
        , scan_offset_(offset)
        , scan_grid_(grid)
        , scan_coords_(3)
    {
        // TODO(lee):
        //  sanity check
        //  net->inputs() == inputs_

        parse_spec(fname);
        setup();
    }

    virtual ~volume_forward_scanner() {}

}; // class volume_forward_scanner

}} // namespace znn::v4