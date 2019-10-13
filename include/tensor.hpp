#pragma once

#include "memory/tensorData.hpp"

#include <type_traits>

namespace math
{

template< typename T, int... SIZES >
class Tensor;

template< typename T, int LEAD_SIZE, int... SIZES >
class Tensor< T, LEAD_SIZE, SIZES... >
{
    static_assert( std::is_arithmetic_v< T >, "Tensor can hold only arithmetic types!" );

private:
    memory::TensorData< Tensor< T,  SIZES... >, LEAD_SIZE > data_;

public:
    std::size_t size() const noexcept 
    {
        return LEAD_SIZE;
    }

    std::size_t dimensionNum() const noexcept
    {
        return sizeof...( SIZES ) + 1;
    }

    auto& operator[]( std::size_t ix ) noexcept
    {
        return data_[ ix ];
    }

    auto const& operator[]( std::size_t ix ) const noexcept
    {
        return data_[ ix ];
    }
};

// scalar tensor
template< typename T >
class Tensor< T >
{
    static_assert( std::is_arithmetic_v< T >, "Tensor can hold only arithmetic types!" );

private:
    memory::TensorData< T > data_;

public:
    Tensor() : data_{ T{} }
    {}

    Tensor( T value ) : data_{ value }
    {}

    T get() const noexcept
    {
        return data_.get();
    }

    std::size_t dimensionNum() const noexcept
    {
        return 0;
    }
};

template< typename T >
using Scalar = Tensor< T >;

} // namespace math