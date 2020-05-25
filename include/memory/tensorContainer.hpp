#pragma once

#include "constants.hpp"
#include "initializers/initializer.hpp"
#include "initializers/zeroInitializer.hpp"

#include <range/v3/view/span.hpp>
#include <range/v3/view/enumerate.hpp>

#include <array>
#include <optional>
#include <cassert>
#include <algorithm>

namespace nn
{

namespace memory
{

template< typename T >
class TensorContainerView
{
private:
    ranges::span< T > data_;

public:
    constexpr TensorContainerView( ranges::span< T > data ) : data_{ data } {}

    constexpr auto getView( std::size_t const start, std::size_t const count = 1 ) const noexcept
    {
        return TensorContainerView< T >{ data_.subspan( start, count ) };
    }

    constexpr auto& operator[]( std::size_t const index ) const noexcept
    {
        return data_[ index ];
    }

    constexpr ranges::span< T const > getSpan() const noexcept
    {
        return data_;
    }

    constexpr ranges::span< T > getSpan() noexcept
    {
        return data_;
    }

    template< typename, TensorSize >
    friend class TensorContainer;
};

namespace
{
    static constexpr TensorSize limit = 10000;
}

template< typename T, TensorSize size = 0 >
class DynamicElement
{
private:
    std::unique_ptr< T[] > data_;

    void copyElements( std::unique_ptr< T[] > const & source, std::unique_ptr< T[] > & destination ) noexcept
    {
        for ( std::size_t i = 0; i < size; ++i )
        {
            destination[ i ] = source[ i ];
        }
    }
public:
    constexpr operator T*() const noexcept
    {
        return data_.get();
    }

    constexpr DynamicElement() : data_{ new T[ size ] }
    {}

    constexpr DynamicElement( DynamicElement< T, size > const & other ) : data_{ new T[ size ] }
    {
        copyElements( other.data_, data_ );
    }

    constexpr DynamicElement< T, size >& operator=( DynamicElement< T, size > const & other ) noexcept
    {
        if ( &other != this )
        {
            copyElements( other.data_, data_ );
        }

        return *this;
    }

    constexpr DynamicElement( DynamicElement< T, size > && other ) : data_{ std::move( other.data_ ) }
    {}

    constexpr DynamicElement< T, size >& operator=( DynamicElement< T, size > && other ) noexcept
    {
        if( &other != this )
        {
            data_ = std::move( other.data_ );
        }

        return *this;
    }

    constexpr auto begin() noexcept
    {
        return std::next( data_.get(), 0 );
    }

    constexpr auto end() noexcept
    {
        return std::next( data_.get(), size );
    }
};
/*
 * Container class for tensors
 */
template< typename T, TensorSize size = 0 >
class TensorContainer
{
private:
    mutable std::conditional_t< ( size > limit ), DynamicElement< T, size >, std::array< T, size > > data_;

    template< typename Container >
    constexpr void assignData( Container const & container ) noexcept
    {
        assert( container.size() == size );

        auto dataIterator{ std::begin( data_ ) };
        for ( auto const & element : container )
        {
            *dataIterator++ = element;
        }
    }

public:
    template< template< typename > class Initializer >
    constexpr TensorContainer
    (
        Initializer< T > const & initializer = nn::initializer::ZeroInitializer< T >{},
        nn::initializer::enableIfInitializer< Initializer< T > > * = 0
    )
    {
        std::transform( std::begin( data_ ), std::end( data_ ), std::begin( data_ ), [ &initializer ]( auto const & ){ return initializer.getValue(); } );
    }

    constexpr TensorContainer( ranges::span< T const > const container ) noexcept
    {
        assignData( container );
    }

    constexpr TensorContainer( std::initializer_list< T > const & initList ) noexcept
    {
        assignData( initList );
    }

    constexpr TensorContainer( TensorContainerView< T > const & other )
    {
        assert( other.data_.size() == size );

        for ( auto const & [ index, element ] : other.data_ | ranges::views::enumerate )
        {
            data_[ index ] = element;
        }
    }

    constexpr TensorContainer& operator=( TensorContainerView< T > const & other ) noexcept
    {
        assert( other.data_.size() == size );

        if ( &other != this )
        {
            for ( auto const & [ index, element ] : other.data_ | ranges::enumerate )
            {
                data_[ index ] = element;
            }
        }

        return this;
    }

    constexpr TensorContainer( TensorContainer< T, size > const & other ) : data_{ other.data_ }
    {}

    constexpr TensorContainer< T, size >& operator=( TensorContainer< T, size > const & other ) noexcept
    {
        if ( &other != this )
        {
            data_ = other.data_;
        }

        return *this;
    }

    constexpr TensorContainer( TensorContainer< T, size > && other ) : data_{ std::move( other.data_ ) }
    {}

    constexpr TensorContainer< T, size >& operator=( TensorContainer< T, size > && other ) noexcept
    {
        if( &other != this )
        {
            data_ = std::move( other.data_ );
        }

        return *this;
    }

    constexpr auto getView( std::size_t const start, std::size_t const count = 1 ) const noexcept
    {
        return TensorContainerView< T >{ ranges::span< T >{ std::begin( data_ ), std::end( data_ ) }.subspan( start, count ) };
    }

    constexpr auto& operator[]( std::size_t const index ) const noexcept
    {
        return data_[ index ];
    }

    constexpr ranges::span< T const > getSpan() const noexcept
    {
        return ranges::span< T const >{ std::cbegin( data_ ), std::cend( data_ ) };
    }

    constexpr ranges::span< T > getSpan() noexcept
    {
        return ranges::span< T >{ std::begin( data_ ), std::end( data_ ) };
    }
};

} // namespace memory

} // namespace nn