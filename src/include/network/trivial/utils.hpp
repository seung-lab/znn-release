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

#include "../bias.hpp"
#include "../filter.hpp"
#include "../../types.hpp"
#include "../../cube/cube.hpp"


namespace znn { namespace v4 {

inline void load_biases( std::vector<std::shared_ptr<bias>> const & bs,
                         std::string const & s )
{
    real const* data = reinterpret_cast<real const *>(s.data());
    ZI_ASSERT((s.size()==bs.size()*sizeof(real))||
              (s.size()==bs.size()*sizeof(real)*2));

    for ( size_t i = 0; i < bs.size(); ++i )
    {
        bs[i]->b() = data[i];
    }

    if ( s.size() == bs.size()*sizeof(real)*2 )
    {
        for ( size_t i = 0; i < bs.size(); ++i )
        {
            bs[i]->momentum_value() = data[i+bs.size()];
        }
    }
}

inline std::string save_biases( std::vector<std::shared_ptr<bias>> const & bs )
{
    real* data = new real[bs.size() * 2];

    for ( size_t i = 0; i < bs.size(); ++i )
    {
        data[i]           = bs[i]->b();
        data[i+bs.size()] = bs[i]->momentum_value();
    }

    std::string ret( reinterpret_cast<char*>(data),
                     bs.size() * sizeof(real) * 2 );

    delete[] data;

    return ret;
}

inline void load_filters( std::vector<std::shared_ptr<filter>> const & fs,
                          const vec3i& s,
                          std::string const & d )
{
    long_t n = s[0] * s[1] * s[2];

    real const* data = reinterpret_cast<real const *>(d.data());

    ZI_ASSERT((d.size()==fs.size()*n*sizeof(real))||
              (d.size()==fs.size()*n*sizeof(real)*2));

    size_t idx = 0;

    for ( size_t i = 0; i < fs.size(); ++i )
    {
        ZI_ASSERT(size(fs[i]->W())==s);
        for ( long_t z = 0; z < s[2]; ++z )
            for ( long_t y = 0; y < s[1]; ++y )
                for ( long_t x = 0; x < s[0]; ++x )
                    fs[i]->W()[x][y][z] = data[idx++];
    }

    if ( d.size() == fs.size()*n*sizeof(real)*2 )
    {
        for ( size_t i = 0; i < fs.size(); ++i )
        {
            //fs[i]->momentum_volume().resize(extents[s[0]][s[1]][s[2]]);
            ZI_ASSERT(size(fs[i]->momentum_volume())==s);
            for ( long_t z = 0; z < s[2]; ++z )
                for ( long_t y = 0; y < s[1]; ++y )
                    for ( long_t x = 0; x < s[0]; ++x )
                        fs[i]->momentum_volume()[x][y][z] = data[idx++];
        }
    }
}


inline std::string save_filters( std::vector<std::shared_ptr<filter>> const & fs,
                                 vec3i const & s )
{
    long_t n = s[0] * s[1] * s[2];

    real* data = new real[fs.size() * n * 2];

    size_t idx = 0;

    for ( size_t i = 0; i < fs.size(); ++i )
    {
        for ( long_t z = 0; z < s[2]; ++z )
            for ( long_t y = 0; y < s[1]; ++y )
                for ( long_t x = 0; x < s[0]; ++x )
                    data[idx++] = fs[i]->W()[x][y][z];
    }

    for ( size_t i = 0; i < fs.size(); ++i )
    {
        for ( long_t z = 0; z < s[2]; ++z )
            for ( long_t y = 0; y < s[1]; ++y )
                for ( long_t x = 0; x < s[0]; ++x )
                    data[idx++] = fs[i]->momentum_volume()[x][y][z];
    }

    std::string ret( reinterpret_cast<char*>(data),
                     fs.size() * n * sizeof(real) * 2 );

    delete[] data;

    return ret;
}




}} // namespace znn::v4
