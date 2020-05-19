#pragma once

#include "shape.hpp"

namespace nn
{

namespace
{
    template< TensorSize... sizes >
    constexpr std::size_t numberOfElementsWithWildcard() noexcept
    {
        std::size_t result{ 1 };
        auto const multiplyIfPositive = [ & ]( TensorSize const size ) noexcept
        {
            if ( size > 0 )
            {
                result *= size;
            }
        };

        ( multiplyIfPositive( sizes ), ... );

        return result;
    }

    constexpr TensorSize correctSize( TensorSize const wildcardValue, TensorSize const size ) noexcept
    {
        return size == -1 ? wildcardValue : size;
    }
}

namespace detail
{

template< std::size_t expectedNumberOfElements, typename Shape, TensorSize... sizes >
struct FillWildcard;

template< std::size_t expectedNumberOfElements, TensorSize... sizes >
struct FillWildcard< expectedNumberOfElements, nn::Shape< sizes... > >
{
    static_assert( expectedNumberOfElements % numberOfElementsWithWildcard< sizes... >() == 0, "Invalid reshape values" );
    static constexpr TensorSize wildcardValue = expectedNumberOfElements / numberOfElementsWithWildcard< sizes... >();
    using shape = nn::Shape< correctSize( wildcardValue, sizes )... >;
};

template< typename OldShape, typename NewShape >
struct ShapeWithWildcardDeducer
{
    static_assert( nn::is_shape_v< OldShape >, "Only Shape allowed" );
    static_assert( nn::is_shape_v< NewShape >, "Only Shape allowed" );

    using shape = typename FillWildcard< OldShape::numberOfElements(), NewShape >::shape;
};

}

}