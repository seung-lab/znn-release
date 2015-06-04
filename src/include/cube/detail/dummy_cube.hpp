#pragma once

#include <memory>

#include "../../types.hpp"

namespace znn { namespace v4 {

template <typename T>
using cube = boost::multi_array<T, 3>;

template <typename T>
using qube = boost::multi_array<T, 4>;

template<typename T>
std::shared_ptr<cube<T>> get_cube(const vec3i& s)
{
    return std::shared_ptr<cube<T>>
        (new cube<T>(extents[s[0]][s[1]][s[2]]));
}

template<typename T>
std::shared_ptr<qube<T>> get_qube(const vec4i& s)
{
    return std::shared_ptr<qube<T>>
        (new qube<T>(extents[s[0]][s[1]][s[2]][s[3]]));
}

}} // namespace znn::v4
