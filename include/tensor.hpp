#pragma once

#include "memory/tensorContainer.hpp"
#include "initializers/initializer.hpp"
#include "initializers/zeroInitializer.hpp"
#include "detail/shapeOperations.hpp"

#include <range/v3/view/span.hpp>

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
    constexpr bool isView = type == TensorType::view || type == TensorType::constView;
}

template< typename TensorElementType, typename TensorShape, TensorType type = TensorType::regular >
class Tensor
{
public:
    using Shape       = TensorShape;
    using ElementType = std::decay_t< TensorElementType >;

    static_assert( nn::is_shape_v< Shape >, "Second argument of tensor should be of class Shape" );
    static_assert( Shape::isValid(), "All shape sizes should be numbers greater than 0." );
    static_assert( std::is_arithmetic_v< ElementType >, "Tensor can hold only arithmetic types!" );
private:
    std::conditional_t< isView< type >, memory::TensorContainerView< ElementType >, memory::TensorContainer< ElementType, Shape::numberOfElements() > > data_;

    template< TensorType TT = type, typename = std::enable_if_t< isView< TT > > >
    constexpr Tensor( memory::TensorContainerView< ElementType > && containerView ) : data_{ std::move( containerView ) }
    {}

public:
    template
    <
        TensorType TT = type,
        template< typename = ElementType > class Initializer = initializer::ZeroInitializer,
        typename = std::enable_if_t< !isView< TT > && initializer::is_initializer_v< Initializer< ElementType > > >
    >
    explicit constexpr Tensor
    (
        Initializer< ElementType > const & initializer = initializer::ZeroInitializer< ElementType >{}
    )
        : data_{ initializer }
    {}

    template< TensorType TT = type, typename = std::enable_if_t< !isView< TT > > >
    explicit constexpr Tensor( ranges::span< ElementType const > const span )
        : data_{ span }
    {}

    template< TensorType TT = type, typename = std::enable_if_t< !isView< TT > > >
    constexpr Tensor( std::initializer_list< ElementType > const initList )
        : data_{ initList }
    {}

    template< TensorType TT = type, TensorType otherTensorType, typename = std::enable_if_t< !isView< TT > > >
    explicit constexpr Tensor( Tensor< ElementType, Shape, otherTensorType > const & other )
        : data_{ other.data_ }
    {}

    template< TensorType TT = type, TensorType otherTensorType, typename = std::enable_if_t< !isView< TT > > >
    constexpr auto & operator=( Tensor< ElementType, Shape, otherTensorType > const & other ) noexcept
    {
        if ( &other != this )
        {
            data_ = other.data_;
        }

        return *this;
    }

    template< TensorType TT = type, typename = std::enable_if_t< !isView< TT > >  >
    constexpr Tensor( Tensor< ElementType, Shape, TensorType::regular > && other  )
        : data_{ std::move( other.data_ ) }
    {}

    template< TensorType TT = type, typename = std::enable_if_t< !isView< TT > > >
    constexpr auto & operator=( Tensor< ElementType, Shape, TensorType::regular > && other ) noexcept
    {
        if ( &other != this )
        {
            data_ = std::move( other );
        }
        return *this;
    }

    [[ nodiscard ]]
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
            return { data_.getView( ix * Shape::stride( 0 ), Shape::stride( 0 ) ) };
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
            return { data_.getView( ix * Shape::stride( 0 ), Shape::stride( 0 ) ) };
        }
    }

    constexpr ranges::span< ElementType const > getAllElementsView() const noexcept
    {
        return data_.getSpan();
    }

    constexpr ranges::span< ElementType > getAllElementsView() noexcept
    {
        return data_.getSpan();
    }

    template<typename, typename, TensorType>
    friend class Tensor;
};

// type trait

template< typename > struct is_tensor : std::false_type {};
template< typename ElementType, typename Shape, TensorType type > struct is_tensor< nn::Tensor< ElementType, Shape, type > > : std::true_type {};

template< typename Tensor >
inline constexpr bool is_tensor_v = is_tensor< Tensor >::value;

template< typename, typename > struct is_same_shape_tensor : std::false_type {};
template< typename ElementType, typename Shape, TensorType tensorType1, TensorType tensorType2 >
struct is_same_shape_tensor< Tensor< ElementType, Shape, tensorType1 >, Tensor< ElementType, Shape, tensorType2 > > : std::true_type{};

template< typename Tensor1, typename Tensor2 >
inline constexpr bool is_same_shape_tensor_v = is_same_shape_tensor< Tensor1, Tensor2 >::value;

} // namespace nn