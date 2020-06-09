#pragma once

#include "activation.hpp"

#include <algorithm>

namespace nn::activations
{

template< typename T >
struct Relu
{
    T operator()( T const & other ) const noexcept
    {
        return std::max( T{ 0 }, other );
    }
};

template< typename T >
struct is_activation< Relu< T > > : std::true_type{};

} // namespace nn::activations