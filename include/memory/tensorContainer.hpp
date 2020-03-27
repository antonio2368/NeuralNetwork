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

/*
 * Container class for tensors
 */
template< typename Tensor, TensorSize SIZE = 0 >
class TensorContainer
{
    using TensorElementType = typename Tensor::ElementType;
private:
    std::array < Tensor, SIZE  > data_;

    void createTensors( nn::initializer::InitializerBase< TensorElementType > const& initializer )
    {
        std::transform( std::begin( data_ ), std::end( data_ ), std::begin( data_ ), [ &initializer ]( auto& ){ return Tensor{ initializer }; } );
    }
public:
    TensorContainer( nn::initializer::InitializerBase< TensorElementType > const& initializer = nn::initializer::ZeroInitializer< TensorElementType >{} )
    {
        createTensors( initializer );
    }

    template< typename Container >
    TensorContainer( Container const & container )
    {
        assert( std::size( container ) == SIZE );
        std::transform( std::begin( container ), std::end( container ), std::begin( data_ ), []( auto const& elem )
        {
            using ContainerValueType = typename Container::value_type;
            if constexpr ( std::is_arithmetic_v< ContainerValueType > )
            {
                return Tensor{ static_cast< TensorElementType >( elem ) };
            }
            else
            {
                return Tensor{ elem };
            }

        });
    }

    /* Const index operator for the tensor container.
     * ix - index of the element
     */
    [[ nodiscard ]]
    Tensor const& operator[]( std::size_t const ix ) const noexcept
    {
        // TODO: add index bounds check
        return data_[ ix ];
    }

    /* Index operator for the tensor container.
     * ix - index of the element
     */
    Tensor& operator[]( std::size_t const ix ) noexcept
    {
        // TODO: add index bounds check
        return data_[ ix ];
    }
};

template< typename Tensor >
class TensorContainer< Tensor, Dynamic >
{
    using TensorElementType = typename Tensor::ElementType;
private:
    std::optional< std::size_t > size_;
    std::vector< Tensor > data_;
public:
    TensorContainer() = default;

    TensorContainer( TensorContainer< Tensor > const& other ) : size_{ other.size_ }
    {
        memcpy( data_, other.data_, size_ );
    }

    template< typename... Args >
    void initialize( Args&&... args )
    {
        assert( size_ );
        assert( data_.capacity() == size_.value() );

        for ( int i = 0; i < size_; ++i )
        {
            data_.emplace_back( std::forward< Args >( args )... );
        }
    }

    void setSize( std::size_t const size ) noexcept
    {
        assert( !size );

        size_.emplace( size );
        data_.resize( size_.value() );
    }

    void resetSize() noexcept
    {
        assert( size_ );
        size_.reset();
        data_.clear();
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
};

template< typename T >
class TensorContainer< T >
{
private:
    T data_;
public:
    TensorContainer( nn::initializer::InitializerBase< T > const& initializer = nn::initializer::ZeroInitializer< T >{} )
    {
        data_ = initializer.getValue();
    }

    TensorContainer( T value ) : data_{ value }
    {}

    [[ nodiscard ]]
    T get() const noexcept
    {
        return data_;
    }

    TensorContainer< T >& operator=( T&& data ) noexcept
    {
        data_ = std::move( data );
        return *this;
    }
};

} // namespace memory

} // namespace nn