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
#pragma once

#include "../convolver.hpp"
#include "../../cube/cube.hpp"

namespace znn { namespace v4 {

template< typename T >
class convolver_constant: public convolver<T>
{
public:
    cube_p<T> forward( ccube_p<T> const & a, ccube_p<T> const & b) override
    {
        ZI_ASSERT(size(b)==vec3i::one);

        cube_p<T> r = get_cube<T>(size(a));
        T c = b.data()[0];

        T const * ap = a.data();
        T * rp = r->data();

        for ( size_t i = 0; i < r.num_elements(); ++i )
            rp[i] = ap[i] * c;

        return r;
    }

    cube_p<T> flipped( ccube_p<T> const & a, ccube_p<T> const & b) override
    {
        ZI_ASSERT(size(a)==size(b));

        cube_p<T> r = get_cube<T>(vec3i::one);
        T& rp = r->data()[0];

        T const * ap = a.data();
        T const * bp = b.data();

        for ( size_t i = 0; i < a.num_elements(); ++i )
            rp += ap[i] * bp[i];

        return r;
    }

    cube_p<T> inverse( ccube_p<T> const & a, ccube_p<T> const & b) override
    {
        return convolver_constant::forward(a,b);
    }
};

}} // namespace znn::v4

#endif
