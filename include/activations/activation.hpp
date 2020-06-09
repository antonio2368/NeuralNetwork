#pragma once

#include <type_traits>

namespace nn::activations
{

template< typename T >
struct is_activation : std::false_type{};

template< typename T >
inline constexpr bool is_activation_v = is_activation< T >::value;

} // namespace nn::activations