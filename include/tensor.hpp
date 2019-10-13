#pragma once

#include "memory/tensorData.hpp"
#include "initializers/initializer.hpp"
#include "initializers/zeroInitializer.hpp"

#include <type_traits>

namespace nn
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
    using ElementType = T;

    Tensor( nn::initializer::InitializerBase< T > && initializer = nn::initializer::ZeroInitializer< T >{} )
        : data_{ std::move( initializer ) }
    {}

    Tensor( nn::initializer::InitializerBase< T > const & initializer )
        : data_{ initializer }
    {}

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
     using ElementType = T;

    Tensor( nn::initializer::InitializerBase< T > && initializer = nn::initializer::ZeroInitializer< T >{} )
        : data_{ std::move( initializer ) }
    {}

    Tensor( nn::initializer::InitializerBase< T > const & initializer )
        : data_{ initializer }
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

    Tensor< T >& operator=( T&& data )
    {
        data_ = std::move( data );
        return *this;
    }

    operator T() const noexcept
    {
        return data_.get();
    }
};

template< typename T >
using Scalar = Tensor< T >;

} // namespace nn