//
// Copyright (C) 2012-2015  Aleksandar Zlateski <zlateski@mit.edu>
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
#include "network/trivial/trivial_fft_network.hpp"
#include "network/trivial/trivial_network.hpp"
#include "network/helpers.hpp"
#include <zi/zargs/zargs.hpp>
#include <fstream>

using namespace znn::v4;

int main(int argc, char** argv)
{
    std::ofstream ofs;
    ofs.open (argv[1], std::ofstream::out | std::ofstream::app);

    std::vector<options> nodes, edges;

    int64_t W = atoi(argv[2]);
    int64_t D = atoi(argv[3]);

    nodes.resize(D+1);
    edges.resize(D);

    nodes[0].push("name", "input").push("type", "input").push("size", 1);

    for ( int64_t i = 0; i < D; ++i )
    {
        edges[i].push("name", i).push("type", "conv").push("init", "uniform")
            .push("size", "5,5,5").push("stride", "1,1,1")
            .push("input", i-1).push("output",i);
        nodes[i+1].push("name",i).push("type","transfer")
            .push("function","rectify_linear").push("size",W);
    }

    edges[0].push("input","input");
    edges[D-1].push("output","output");
    nodes[D].push("name","output");
    nodes[D].push("size","1");

    int64_t x = 9;
    int64_t y = 9;
    int64_t z = 9;

    if ( argc >= 7 )
    {
        x = atoi(argv[4]);
        y = atoi(argv[5]);
        z = atoi(argv[6]);
    }

    size_t warmup = 3;

    if ( argc >= 8 )
    {
        warmup = atoi(argv[7]);
    }

    size_t nrnds = 31;

    if ( argc >= 9 )
    {
        nrnds = atoi(argv[8]);
    }


    size_t max_threads = 240;

    if ( argc >= 10 )
    {
        max_threads = atoi(argv[0]);
    }

    std::vector<double> speeds(max_threads+1);

    for ( int i = 1; i <= max_threads; ++i )
    {
        auto res = parallel_network::network::speed_test
            (nodes, edges, {z,y,x}, i, nrnds, warmup);

        speeds[i] = res.first;

        ofs << W << ", " << D << ", " << i << ", "
            << res.first << ", " << res.second << ", "
            << (speeds[1] / speeds[i] ) << ";" << std::endl;

        std::cout << W << ", " << D << ", " << i << ", "
                  << res.first << ", " << res.second
                  << " ( " << ( res.second * 100  / res.first ) << "% )"
                  << ";" << std::endl
                  << "____SPEEDUP: " << ( speeds[1] / speeds[i] ) << std::endl;
    }
}
