#pragma once

#include "memory/tensorContainer.hpp"
#include "initializers/initializer.hpp"
#include "initializers/zeroInitializer.hpp"
#include "typeTraits.hpp"
#include "shape.hpp"

#include <type_traits>

namespace nn
{

template< typename T, typename Shape = Shape<> >
class Tensor
{
    static_assert( nn::is_shape_v< Shape >, "Second argument of tensor should be of class Shape" );
    static_assert( std::is_arithmetic_v< T >, "Tensor can hold only arithmetic types!" );

private:
    memory::TensorContainer< Tensor< T,  typename Shape::SubShape >, Shape::size( 0 ) > data_;

public:
    using ElementType = T;

    using TensorShape = Shape;

    Tensor( nn::initializer::InitializerBase< T > const & initializer = nn::initializer::ZeroInitializer< T >{} )
        : data_{ initializer }
    {}

    template< typename Container >
    Tensor( Container const& container )
        : data_{ container }
    {}

    std::size_t size() const noexcept
    {
        return Shape::size( 0 );
    }

    static constexpr std::size_t dimensions() noexcept
    {
        return Shape::dimensions();
    }

    auto& operator[]( std::size_t const ix ) noexcept
    {
        assert( ix < Shape::size( 0 ) );
        return data_[ ix ];
    }

    auto const& operator[]( std::size_t const ix ) const noexcept
    {
        assert( ix < Shape::size( 0 ) );
        return data_[ ix ];
    }
};

// scalar tensor
template< typename T >
class Tensor< T >
{
    static_assert( std::is_arithmetic_v< T >, "Tensor can hold only arithmetic types!" );

private:
    memory::TensorContainer< T > data_;

public:
     using ElementType = T;

    Tensor( nn::initializer::InitializerBase< T > const & initializer = nn::initializer::ZeroInitializer< T >{} )
        : data_{ initializer }
    {}

    Tensor( T value ) : data_{ value }
    {}

    T get() const noexcept
    {
        return data_.get();
    }

    static constexpr std::size_t dimensions() noexcept
    {
        return Shape<>::dimensions();
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

// type trait

template< typename > struct is_tensor : std::false_type {};
template< typename T, typename Shape > struct is_tensor< nn::Tensor< T, Shape > > : std::true_type {};
template< typename T > struct is_tensor< nn::Tensor< T > > : std::true_type {};

template< typename T >
inline constexpr bool is_tensor_v = is_tensor< T >::value;


} // namespace nn