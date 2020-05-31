#pragma once

#include "../shape.hpp"

namespace nn
{

namespace detail
{

constexpr TensorSize correctSize( TensorSize const wildcardValue, TensorSize const size ) noexcept
{
    return size == shapeWildcardSize ? wildcardValue : size;
}

template< std::size_t, typename >
struct FillWildcard;

template< std::size_t expectedNumberOfElements, TensorSize... sizes >
struct FillWildcard< expectedNumberOfElements, nn::Shape< sizes... > >
{
private:
    static constexpr std::size_t currentNumberOfElements = nn::Shape< sizes... >::numberOfElements();
    static_assert( expectedNumberOfElements % currentNumberOfElements == 0, "Invalid reshape values" );
    static constexpr TensorSize wildcardValue = expectedNumberOfElements / currentNumberOfElements;
public:
    using shape = nn::Shape< correctSize( wildcardValue, sizes )... >;
};

template< typename OldShape, typename NewShape >
struct ShapeWithWildcardDeducer
{
    static_assert( nn::is_shape_v< OldShape > && OldShape::isValid()                 , "Invalid old shape" );
    static_assert( nn::is_shape_v< NewShape > && NewShape::template isValid< true >(), "Invalid new shape" );

    using shape = typename FillWildcard< OldShape::numberOfElements(), NewShape >::shape;
};

} // namespace detail

} // namespace nn