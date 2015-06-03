template <typename T>
using cube = boost::multi_array<T, 3, typename std::conditional<
                                          needs_fft_allocator<T>::value,
                                          cube_allocator<T>,
                                          std::allocator<T>>::type >;

template <typename T>
using qube = boost::multi_array<T, 4, typename std::conditional<
                                          needs_fft_allocator<T>::value,
                                          cube_allocator<T>,
                                          std::allocator<T>>::type >;

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
