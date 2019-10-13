#pragma once

#include "constants.hpp"
#include "initializers/initializer.hpp"
#include "initializers/zeroInitializer.hpp"

#include <array>
#include <optional>
#include <cassert>
#include <algorithm>

namespace nn
{

namespace memory
{

template< typename Tensor, int SIZE = 0 >
class TensorData
{
    using TensorElementType = typename Tensor::ElementType;
private:
    std::array < Tensor, SIZE  > data_;

    void createTensors( nn::initializer::InitializerBase< TensorElementType > const& initializer )
    {
        std::transform( data_.begin(), data_.end(), data_.begin(), [ &initializer ]( auto& ){ return Tensor{ initializer }; } );
    }
public:
    TensorData( nn::initializer::InitializerBase< TensorElementType >&& initializer = nn::initializer::ZeroInitializer< TensorElementType >{} )
    {
        createTensors( initializer );
    }

    TensorData( nn::initializer::InitializerBase< TensorElementType > const& initializer )
    {
        createTensors( initializer );
    }

    [[ nodiscard ]]
    Tensor const& operator[]( std::size_t const ix ) const noexcept
    {
        return data_[ ix ];
    }

    Tensor& operator[]( std::size_t const ix ) noexcept
    {
        return data_[ ix ];
    }
};

template< typename Tensor >
class TensorData< Tensor, Dynamic >
{
private:
    std::optional< std::size_t > size_;
    Tensor* data_ = nullptr;
public:
    TensorData() = default;

    TensorData( TensorData< Tensor > const& other ) : size_{ other.size_ }
    {
        memcpy( data_, other.data_, size_ );
    }

    void setSize( std::size_t const size ) noexcept 
    {
        assert( !size );

        size_.emplace( size );

        data_ = new Tensor[ size_.value() ];
    }

    void resetSize() noexcept
    {
        assert( size );
        size_.reset();

        assert( data_ );
        delete [] data_;
        data_ = nullptr;
    }

    std::optional< std::size_t > size() const noexcept
    {
        return size_;
    }

    [[ nodiscard ]]
    Tensor const& operator[]( std::size_t const ix ) const noexcept
    {
        assert( size_ );

        if ( ix < size_ )
        {
            return data_[ ix ];
        }
    }

    Tensor& operator[]( std::size_t const ix ) noexcept
    {
        assert( size_ );

        if ( ix < size_ )
        {
            return data_[ ix ];
        }
    }

    ~TensorData()
    {
        if ( data_ )
        {
            delete [] data_;
            data_ = nullptr;
        }
    }

};

template< typename T >
class TensorData< T >
{
private:
    T data_;
public:
    TensorData( nn::initializer::InitializerBase< T >&& initializer = nn::initializer::ZeroInitializer< T >{} )
    {
        data_ = initializer.getValue();
    }

    TensorData( nn::initializer::InitializerBase< T > const& initializer )
    {
        data_ = initializer.getValue();
    }

    TensorData( T value ) : data_{ value }
    {}

    [[ nodiscard ]]
    T get() const noexcept
    {
        return data_;
    }

    TensorData< T >& operator=( T&& data ) noexcept
    {
        data_ = std::move( data );
        return *this;
    }
};

} // namespace memory

} // namespace nn