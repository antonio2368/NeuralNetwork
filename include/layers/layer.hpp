#pragma once

#include "../tensor.hpp"
#include "../shape.hpp"
#include "../typeTraits.hpp"

#include <type_traits>
#include <iostream>

namespace nn
{

template< typename T,
          typename InputShape,
          typename OutputShape
        >
class Layer
{
    static_assert( nn::is_shape_v< InputShape >, "Input shape not valid" );
    static_assert( nn::is_shape_v< OutputShape >, "Output shape not valid" );

public:
    constexpr auto const& inputShape() const noexcept
    {
        return InputShape::shape();
    }

    constexpr auto const& outputShape() const noexcept
    {
        return OutputShape::shape();
    }
};

} // namespace nn