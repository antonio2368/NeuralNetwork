#pragma once

#include "memory/tensorData.hpp"

#include <type_traits>

namespace math
{

template< typename T, int LEAD_SIZE, int... SIZES >
class Tensor
{
    static_assert( std::is_arithmetic_v< T >, "Tensor can hold only arithmetic types!" );

private:
    memory::TensorData< T, LEAD_SIZE, SIZES... > data_;

public:
    std::size_t size() const noexcept 
    {
        return LEAD_SIZE;
    }

    std::size_t dimensionNum() const noexcept
    {
        return sizeof...( SIZES ) + 1;
    }

    auto& operator[]( std::size_t ix ) const noexcept
    {
        return data_[ ix ];
    }
};

} // namespace math