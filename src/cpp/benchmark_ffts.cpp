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

using namespace znn::v4;

int main(int argc, char** argv)
{
    int64_t x = 9;
    int64_t y = 9;
    int64_t z = 9;

    if ( argc >= 4 )
    {
        x = atoi(argv[1]);
        y = atoi(argv[2]);
        z = atoi(argv[3]);
    }

    size_t tc = 10;

    if ( argc == 5 )
    {
        tc = atoi(argv[4]);
    }

    auto v = get_cube<real>(vec3i(x,y,z));

    for ( int i = 0; i < v->num_elements(); ++i )
        v->data()[i] = i;

    auto f = fftw::forward(std::move(v));
    v = fftw::backward(std::move(f), vec3i(x,y,z));
    *v /= (x*y*z);

    zi::wall_timer wt;
    wt.reset();

    for ( size_t i = 0; i < tc; ++i )
    {
        f = fftw::forward(std::move(v));
        v = fftw::backward(std::move(f), vec3i(x,y,z));
        *v /= (x*y*z);
    }

    std::cout << "Elapsed: " << wt.elapsed<double>() << std::endl;
    std::cout << "Sum: " << sum(*v) << std::endl;

}
