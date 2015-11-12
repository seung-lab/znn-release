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
#include "convolution/convolver_constant.hpp"

using namespace znn::v4;

int main(int argc, char** argv)
{
    uniform_init init(0.001);
    auto v = get_cube<real>(vec3i(3,3,3));
    auto f = get_cube<real>(vec3i(1,1,1));

    init.initialize(v);
    init.initialize(f);

    convolver_constant<real> cc;

    auto z = cc.forward(v,f);

    std::cout << (*z) << std::endl;

    // generate
}
