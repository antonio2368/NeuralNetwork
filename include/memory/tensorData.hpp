#pragma once

#include "constants.hpp"

#include <array>
#include <optional>
#include <cassert>

namespace math
{

namespace memory
{

template < typename T, int SIZE = 0 >
class TensorData
{
private:
    std::array < T, SIZE  > data_;
public:
    TensorData() : data_{ T{} }
    {}

    [[ nodiscard ]]
    T const& operator[]( std::size_t const ix ) const noexcept
    {
        return data_[ ix ];
    }

    T& operator[]( std::size_t const ix ) noexcept
    {
        return data_[ ix ];
    }
};

template< typename T >
class TensorData< T, Dynamic >
{
private:
    std::optional< std::size_t > size_;
    T* data_ = nullptr;
public:
    TensorData() = default;

    TensorData( TensorData< T > const& other ) : size_{ other.size_ }
    {
        memcpy( data_, other.data_, size_ );
    }

    void setSize( std::size_t const size ) noexcept 
    {
        assert( !size );

        size_.emplace( size );

        data_ = new T[ size_.value() ];
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
    T const& operator[]( std::size_t const ix ) const noexcept
    {
        assert( size_ );

        if ( ix < size_ )
        {
            return data_[ ix ];
        }
    }

    T& operator[]( std::size_t const ix ) noexcept
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
    TensorData() = default;

    TensorData( T value ) : data_{ value }
    {}

    [[ nodiscard ]]
    T get() const noexcept
    {
        return data_;
    }

    TensorData< T >& operator=( T const data ) noexcept
    {
        data_ = data;
        return *this;
    }
};

} // namespace memory

} // namespace math