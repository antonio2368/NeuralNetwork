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
        return size == shapeWildcardSize ? wildcardValue : size;
    }

    template< TensorSize... sizes >
    constexpr bool sizesHaveWildcard() noexcept
    {
        auto const isWildcard = []( auto const size ) noexcept
        {
            return size == shapeWildcardSize;
        };

        return ( isWildcard( sizes ) || ... );
    }

    template< typename >
    struct HasWildcard : std::false_type
    {};

    template< TensorSize... sizes >
    struct HasWildcard< nn::Shape< sizes... > > : std::conditional_t< sizesHaveWildcard< sizes... >(), std::true_type, std::false_type >
    {};

    template< std::size_t, typename >
    struct FillWildcard;

    template< std::size_t expectedNumberOfElements, TensorSize... sizes >
    struct FillWildcard< expectedNumberOfElements, nn::Shape< sizes... > >
    {
        static_assert( expectedNumberOfElements % numberOfElementsWithWildcard< sizes... >() == 0, "Invalid reshape values" );
        static constexpr TensorSize wildcardValue = expectedNumberOfElements / numberOfElementsWithWildcard< sizes... >();
        using shape = nn::Shape< correctSize( wildcardValue, sizes )... >;
    };
}

namespace detail
{

template< typename T >
inline constexpr bool hasWildcard = HasWildcard< T >::value;

constexpr bool isValidShapeSize( TensorSize const size ) noexcept
{
    return size > 0;
}

template< typename OldShape, typename NewShape >
struct ShapeWithWildcardDeducer
{
    static_assert( nn::is_shape_v< OldShape > && OldShape::isValid()                 , "Invalid old shape" );
    static_assert( nn::is_shape_v< NewShape > && NewShape::template isValid< true >(), "Invalid new shape" );

    using shape = typename FillWildcard< OldShape::numberOfElements(), NewShape >::shape;
};

} // namespace detail

} // namespace nn