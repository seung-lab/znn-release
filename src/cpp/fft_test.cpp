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

void test_single_fft( vec3i const & size )
{
    auto a1 = get_cube<real>(size);

    auto l = a1->num_elements();

    for ( size_t i = 0; i < l; ++i )
    {
        a1->data()[i] = i % 3;
    }

    auto a2 = get_copy(*a1);

    auto f = fftw::transformer(size);

    auto c = f.forward(std::move(a2));
    a2 = f.backward(std::move(c));

    *a2 /= (size[0] * size[1] * size[2]);

    *a1 -= *a2;

    //std::cout << *a1 << "\n\n" << *a2 << "\n\n";

    real max = 0;
    for ( size_t i = 0; i < l; ++i )
    {
        max = std::max(max, std::abs(a1->data()[i]));
    }

    std::cout << size << ' ' << max << std::endl;
}

int main()
{
    for ( int i = 1; i < 10; ++i )
        for ( int j = 1; j < 10; ++j )
            for ( int k = 1; k < 10; ++k )
                test_single_fft(vec3i(i,j,k));
}
