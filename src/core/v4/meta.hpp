#pragma once

#include <type_traits>

namespace znn { namespace v4 {

template<typename T>
struct identity { typedef T type; };

template<typename T>
using identity_t = typename identity<T>::type;

template<bool B>
using bool_constant = std::integral_constant<bool,B>;

template<class...>
struct void_t_helper_struct { typedef void type; };

template<class... Ts>
using void_t = typename void_t_helper_struct<Ts...>::type;

template<bool B, class T = void>
using if_t = typename std::enable_if<B,T>::type;

}} // namespace znn::v4
