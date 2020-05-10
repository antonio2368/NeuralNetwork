#pragma once

#include "constants.hpp"

#include <array>
#include <type_traits>

namespace nn
{

template< TensorSize... SIZES >
class Shape;

template< TensorSize LEAD_SIZE, TensorSize... SUBSIZES >
class Shape< LEAD_SIZE, SUBSIZES... >
{
private:
    static constexpr std::array< TensorSize, sizeof...( SUBSIZES ) + 1 > shape_{ { LEAD_SIZE, SUBSIZES... } };

public:
    using SubShape = Shape< SUBSIZES... >;

    static constexpr std::size_t numberOfElements() noexcept
    {
        TensorSize numberOfElements{ 1 };
        for ( auto const & size : shape_ )
        {
            numberOfElements *= size;
        }
        return numberOfElements;
    }

    static inline constexpr auto const& shape() noexcept
    {
        return shape_;
    }

    static inline constexpr TensorSize size( std::size_t const index ) noexcept
    {
        return shape_[ index ];
    }

    static constexpr std::size_t dimensions() noexcept
    {
        return sizeof...( SUBSIZES ) + 1;
    }
};

template< typename > struct is_shape : std::false_type {};
template< TensorSize... SIZES > struct is_shape< nn::Shape< SIZES... > > : std::true_type {};

template< typename T >
inline constexpr bool is_shape_v = is_shape< T >::value;

} // namespace nn