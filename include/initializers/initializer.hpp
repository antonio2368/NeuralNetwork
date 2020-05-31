#pragma once

#include <type_traits>

namespace nn::initializer
{

template< typename T >
struct is_initializer : std::false_type{};

template< typename T >
inline constexpr bool is_initializer_v = is_initializer< T >::value;

template< typename T >
using enableIfInitializer = std::enable_if_t< is_initializer_v< T > >;

} // namespace nn::initializer