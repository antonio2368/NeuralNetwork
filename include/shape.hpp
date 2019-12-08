#pragma once

#include <type_traits>


namespace nn
{


template< int... SIZES >
class Shape;

template< int LEAD_SIZE, int... SUBSIZES >
class Shape< LEAD_SIZE, SUBSIZES... >
{
private:
    static constexpr std::array< int, sizeof...( SUBSIZES ) + 1 > shape_{ { LEAD_SIZE, SUBSIZES... } };

public:
    using SubShape = Shape< SUBSIZES... >;

    static inline constexpr auto const& getShape() noexcept
    {
        return shape_;
    }

    static inline constexpr int getSize( std::size_t index ) noexcept
    {
        return shape_[ index ];
    }

    static constexpr int dimensions() noexcept
    {
        return sizeof...( SUBSIZES ) + 1;
    }
};

template<>
class Shape<>
{
    static inline constexpr int dimensions() noexcept
    {
        return 0;
    }

};

} // namespace nn
