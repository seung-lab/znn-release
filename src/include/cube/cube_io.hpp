//
// Copyright (C) 2012-2015  Aleksandar Zlateski <zlateski@mit.edu>
//                    2015  Kisuk Lee           <kisuklee@mit.edu>
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

namespace znn { namespace v4 {

template<typename F, typename T>
inline cube_p<T> read( std::string const & fname, vec3i const & sz )
{
    FILE* fvol = fopen(fname.c_str(), "r");

    STRONG_ASSERT(fvol);

    auto ret = get_cube<T>(sz);
    F v;

    for ( long_t z = 0; z < sz[0]; ++z )
        for ( long_t y = 0; y < sz[1]; ++y )
            for ( long_t x = 0; x < sz[2]; ++x )
            {
                static_cast<void>(fread(&v, sizeof(F), 1, fvol));
                (*ret)[z][y][x] = static_cast<T>(v);
            }

    fclose(fvol);

    return ret;
}

inline bool export_size_info( std::string const & fname,
                              vec3i const & sz, size_t n = 0 )
{
    std::string ssz = fname + ".size";

    FILE* fsz = fopen(ssz.c_str(), "w");

    uint32_t v;

    v = static_cast<uint32_t>(sz[2]); // x
    static_cast<void>(fwrite(&v, sizeof(uint32_t), 1, fsz));

    v = static_cast<uint32_t>(sz[1]); // y
    static_cast<void>(fwrite(&v, sizeof(uint32_t), 1, fsz));

    v = static_cast<uint32_t>(sz[0]); // z
    static_cast<void>(fwrite(&v, sizeof(uint32_t), 1, fsz));

    if ( n )
    {
        v = static_cast<uint32_t>(n);
        static_cast<void>(fwrite(&v, sizeof(uint32_t), 1, fsz));
    }

    fclose(fsz);

    return true;
}

template<typename F, typename T>
inline bool write( std::string const & fname, cube_p<T> vol )
{
    FILE* fvol = fopen(fname.c_str(), "w");

    STRONG_ASSERT(fvol);

    vec3i const & sz = size(*vol);
    F v;

    for ( long_t z = 0; z < sz[0]; ++z )
        for ( long_t y = 0; y < sz[1]; ++y )
            for ( long_t x = 0; x < sz[2]; ++x )
            {
                v = static_cast<T>((*vol)[z][y][x]);
                static_cast<void>(fwrite(&v, sizeof(F), 1, fvol));
            }

    fclose(fvol);

    return export_size_info(fname, sz);
}

template<typename F, typename T>
inline bool write_tensor( std::string const & fname,
                          std::vector<cube_p<T>> vols )
{
    ZI_ASSERT(vols.size()>0);

    FILE* fvol = fopen(fname.c_str(), "w");

    STRONG_ASSERT(fvol);

    F v;

    vec3i const & sz = size(*vols[0]);
    for ( auto& vol: vols )
    {
        ZI_ASSERT(size(*vol)==sz);
        for ( long_t z = 0; z < sz[0]; ++z )
            for ( long_t y = 0; y < sz[1]; ++y )
                for ( long_t x = 0; x < sz[2]; ++x )
                {
                    v = static_cast<T>((*vol)[z][y][x]);
                    static_cast<void>(fwrite(&v, sizeof(F), 1, fvol));
                }
    }

    fclose(fvol);

    return export_size_info(fname, sz, vols.size());
}

}} // namespace znn::v4