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

    template< bool allowWildcard = false, typename Validator >
    static constexpr bool isValidSize( Validator && validator ) noexcept
    {
        bool wildcardFound = false;
        for ( auto const & size : shape_ )
        {
            if constexpr ( allowWildcard )
            {
                if ( size == shapeWildcardSize )
                {
                    if ( wildcardFound )
                    {
                        return false;
                    }
                    wildcardFound = true;
                    continue;
                }
            }
            if ( !validator( size ) )
            {
                return false;
            }
        }

        return true;
    }

public:
    using SubShape = Shape< subSizes... >;

    Shape()=delete;

    template< bool allowWildcard = false, typename Validator >
    static constexpr bool isValid( Validator && validator ) noexcept
    {
        return isValidSize< allowWildcard >( std::forward< Validator >( validator ) );
    }

    template< bool allowWildcard = false >
    static constexpr bool isValid() noexcept
    {
        return isValidSize< allowWildcard >( []( auto const size ) noexcept { return size > 0; } );
    }

    static constexpr bool hasWildcard() noexcept
    {
        for ( auto const size : shape_ )
        {
            if ( size == shapeWildcardSize )
            {
                return true;
            }
        }

        return false;
    }

    static constexpr std::size_t numberOfElements() noexcept
    {
        std::size_t numberOfElements{ 1 };
        for ( auto const & size : shape_ )
        {
            if ( size > 0 )
            {
                numberOfElements *= size;
            }
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