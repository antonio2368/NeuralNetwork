#pragma once

#include "memory/tensorContainer.hpp"
#include "initializers/initializer.hpp"
#include "initializers/zeroInitializer.hpp"
#include "typeTraits.hpp"
#include "shape.hpp"

#include <type_traits>

namespace nn
{

enum class TensorType : std::uint8_t
{
    regular = 0,
    view,
    constView
};

namespace
{
    template< TensorType type >
    constexpr bool isView() noexcept
    {
        return type == TensorType::view || type == TensorType::constView;
    }
}

template< typename TensorElementType, typename TensorShape, TensorType type = TensorType::regular >
class Tensor
{
public:
    using Shape       = TensorShape;
    using ElementType = TensorElementType;

    static_assert( nn::is_shape_v< Shape >, "Second argument of tensor should be of class Shape" );
    static_assert( std::is_arithmetic_v< ElementType >, "Tensor can hold only arithmetic types!" );
private:
    std::conditional_t< isView< type >(), memory::TensorContainerView< ElementType >, memory::TensorContainer< ElementType, Shape::numberOfElements() > > data_;

    template< typename ET = ElementType, TensorType TT = type >
    constexpr Tensor( memory::TensorContainerView< ET > && containerView, std::enable_if_t< isView< TT >() > * = 0 ) : data_{ std::move( containerView ) }
    {}

public:
    template< typename ET = ElementType, TensorType TT = type >
    constexpr Tensor( nn::initializer::InitializerBase< ET > const & initializer = nn::initializer::ZeroInitializer< ET >{}, std::enable_if_t< !isView< TT >() > * = 0 )
        : data_{ initializer }
    {}

    template< typename Container, TensorType TT = type >
    constexpr Tensor( Container const & container, std::enable_if_t< !isView< TT >() > * = 0 )
        : data_{ container }
    {}

    template< typename ET = ElementType, TensorType TT = type >
    constexpr Tensor( std::initializer_list< ET > const initList, std::enable_if_t< !isView< TT >() > * = 0 )
        : data_{ initList }
    {}

    template< TensorType TT = type, TensorType otherTensorType >
    constexpr Tensor( Tensor< ElementType, Shape, otherTensorType > const & other, std::enable_if_t< !isView< TT >() > * = 0 )
        : data_{ other.data_ }
    {}

    template< TensorType TT = type, TensorType otherTensorType >
    constexpr
    std::enable_if_t< !isView< TT >(), Tensor< ElementType, Shape, TensorType::regular > > operator=( Tensor< ElementType, Shape, otherTensorType > const & other ) noexcept
    {
        if ( &other != this )
        {
            data_ = other.data_;
        }

        return *this;
    }

    template< TensorType TT = type  >
    constexpr Tensor( Tensor< ElementType, Shape, TensorType::regular > && other, std::enable_if_t< !isView< TT >() > * = 0  )
        : data_{ std::move( other.data_ ) }
    {}

    template< TensorType TT = type >
    constexpr
    std::enable_if_t< !isView< TT >(), Tensor< ElementType, Shape, TensorType::regular > > operator=( Tensor< ElementType, Shape, TensorType::regular > && other ) noexcept
    {
        if ( &other != this )
        {
            data_ = std::move( other );
        }
        return *this;
    }

    constexpr std::size_t size() const noexcept
    {
        return Shape::size( 0 );
    }

    static constexpr std::size_t dimensions() noexcept
    {
        return Shape::dimensions();
    }

    constexpr
    std::conditional_t
    <
        Shape::dimensions() == 1,
        std::conditional_t< type == TensorType::view, ElementType &, ElementType const & >,
        Tensor< ElementType, typename Shape::SubShape, type != TensorType::view ? TensorType::constView : TensorType::view >
    >
    operator[]( std::size_t const ix ) const noexcept
    {
        assert( ix < Shape::size( 0 ) );
        if constexpr ( Shape::dimensions() == 1 )
        {
            return data_[ ix ];
        }
        else
        {
            return { data_.getView( ix * Shape::size( 1 ), Shape::size( 1 ) ) };
        }

    }

    constexpr
    std::conditional_t
    <
        Shape::dimensions() == 1,
        std::conditional_t< type == TensorType::constView, ElementType const &, ElementType & >,
        Tensor< ElementType, typename Shape::SubShape, type == TensorType::constView ? TensorType::constView : TensorType::view >
    >
    operator[]( std::size_t const ix ) noexcept
    {
        assert( ix < Shape::size( 0 ) );
        if constexpr ( Shape::dimensions() == 1 )
        {
            return data_[ ix ];
        }
        else
        {
            return { data_.getView( ix * Shape::size( 1 ), Shape::size( 1 ) ) };
        }
    }

    template<typename, typename, TensorType>
    friend class Tensor;
};

// type trait

template< typename > struct is_tensor : std::false_type {};
template< typename ElementType, typename Shape, TensorType type > struct is_tensor< nn::Tensor< ElementType, Shape, type > > : std::true_type {};

template< typename ElementType >
inline constexpr bool is_tensor_v = is_tensor< ElementType >::value;


} // namespace nn