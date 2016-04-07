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
#include "network/parallel/network.hpp"
#include "front_end/volume_forward_scanner.hpp"
#include "cube/cube_io.hpp"

#include <iostream>

using namespace znn::v4;

int main(int argc, char** argv)
{
    // ---------------------------------------------------------------
    // parse option
    // ---------------------------------------------------------------
    options op;
    std::string fname(argv[1]);
    parse_option_file(op, fname);

    // ---------------------------------------------------------------
    // construct network
    // ---------------------------------------------------------------
    std::vector<options> nodes;
    std::vector<options> edges;

    auto net_spec = op.require_as<std::string>("net_spec");
    parse_net_file(nodes, edges, net_spec);
    parallel_network::network::force_fft(edges);

    auto outsz = op.require_as<ovec3i>("outsz");
    auto tc = op.require_as<size_t>("n_threads");
    parallel_network::network net(nodes,edges,outsz,tc);

    // ---------------------------------------------------------------
    // load input
    // ---------------------------------------------------------------
    auto ipath = op.require_as<std::string>("input_path");
    auto isize = op.require_as<ovec3i>("input_size");
    auto input = read<real,real>(ipath, isize);

    // ---------------------------------------------------------------
    // preprocess input
    // ---------------------------------------------------------------
    input = mirror_boundary(*input, net.fov());

    // ---------------------------------------------------------------
    // construct volume_dataset
    // ---------------------------------------------------------------
    volume_dataset<real> data;
    data.add_data("input",input);
    auto data_p = std::shared_ptr<volume_dataset<real>>(&data);

    // ---------------------------------------------------------------
    // construct volume_forward_scanner
    // ---------------------------------------------------------------
    auto offset = op.optional_as<ovec3i>("scan_offset","0,0,0");
    auto grid = op.optional_as<ovec3i>("scan_grid","0,0,0");
    auto scan_spec = op.optional_as<std::string>("scan_spec","");
    volume_forward_scanner<real> scanner(&net, data_p, scan_spec, offset, grid);

    // ---------------------------------------------------------------
    // forward scan
    // ---------------------------------------------------------------
    scanner.scan();

    // ---------------------------------------------------------------
    // save results
    // ---------------------------------------------------------------
    auto spath = op.require_as<std::string>("save_path");
    auto outputs = scanner.outputs();

    zi::wall_timer wt;
    for ( auto& o: outputs )
    {
        auto& name = o.first;
        auto oname = spath + name + ".bin";

        std::cout << "Writing [" << name << "]...";
        wt.reset();
        write_tensor<real,real>(oname, o.second);
        std::cout << "done. (elapsed: " << wt.elapsed<double>() << ")\n";
    }
}
