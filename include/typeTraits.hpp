#pragma once

#include "shape.hpp"

#include <type_traits>
#include <tuple>

template< typename > struct is_tuple : std::false_type {};
template< typename... T > struct is_tuple< std::tuple< T... > > : std::true_type {};

template< typename T >
inline constexpr bool is_tuple_v = is_tuple< T >::value;

template< typename > struct is_shape : std::false_type {};
template< int... SIZES > struct is_shape< nn::Shape< SIZES... > > : std::true_type {};

template< typename T >
inline constexpr bool is_shape_v = is_shape< T >::value;