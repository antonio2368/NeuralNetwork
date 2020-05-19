#pragma once

#include "constants.hpp"

#include <array>
#include <type_traits>

namespace nn
{

template< TensorSize... sizes >
class Shape;

template< TensorSize leadSize, TensorSize... subSizes >
class Shape< leadSize, subSizes... >
{
private:
    static constexpr std::size_t numberOfSizes() noexcept { return sizeof...( subSizes ) + 1; }
    static constexpr std::array< TensorSize, numberOfSizes() > shape_{ { leadSize, subSizes... } };
public:
    using SubShape = Shape< subSizes... >;

    Shape()=delete;

    template< typename Validator >
    static constexpr std::size_t isValid( Validator && validator ) noexcept
    {
        for ( auto const & size : shape_ )
        {
            if ( !validator( size ) )
            {
                return false;
            }
        }

        return true;
    }

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
        return numberOfSizes();
    }

    static constexpr std::size_t stride( std::size_t const index ) noexcept
    {
        std::size_t currentIndex{ numberOfSizes() - 1 };
        std::size_t currentStride{ 1u };

        while ( currentIndex > index )
        {
            currentStride *= shape_[ currentIndex ];
            --currentIndex;
        }

        return currentStride;
    }
};

template< typename > struct is_shape : std::false_type {};
template< TensorSize... sizes > struct is_shape< nn::Shape< sizes... > > : std::true_type {};

template< typename T >
inline constexpr bool is_shape_v = is_shape< T >::value;

} // namespace nn