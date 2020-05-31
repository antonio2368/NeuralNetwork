#pragma once

#include "tensor.hpp"
#include "detail/shapeOperations.hpp"

#include <range/v3/view/zip.hpp>
#include <range/v3/view/transform.hpp>
#include <range/v3/algorithm/move.hpp>

#include <optional>
#include <algorithm>
#include <numeric>
#include <type_traits>

namespace nn
{

namespace detail
{

template< typename T, bool isConst >
using ConditionalConst = std::conditional_t< isConst, std::decay_t< T >, std::decay_t< T > const >;

template< typename T >
using enableIfTensor = std::enable_if_t< nn::is_tensor_v< std::decay_t< T > > >;

template< bool inPlace = false, typename BinaryOp, typename Tensor, typename = enableIfTensor< Tensor > >
constexpr auto applyElementwiseOperation
(
    ConditionalConst< Tensor, inPlace > & first,
    Tensor const & second,
    BinaryOp && operation
)
{
    auto const applyElementwise = [ & ]( auto const & firstTensor, auto const & secondTensor, auto & resultTensor )
    {
        ranges::move( ranges::views::zip_with( operation, firstTensor.getAllElementsView(), secondTensor.getAllElementsView() ), resultTensor.getAllElementsView().data() );
    };

    if constexpr ( inPlace )
    {
        applyElementwise( first, second, first );
    }
    else
    {
        nn::Tensor< typename Tensor::ElementType, typename Tensor::Shape, TensorType::regular > result;
        applyElementwise( first, second, result );
        return result;
    }
}

// broadcast
template
<
    bool inPlace = false,
    typename Tensor1,
    typename Tensor2,
    typename BinaryOp,
    typename = enableIfTensor< Tensor2 >
>
constexpr auto applyElementwiseOperation
(
    Tensor1 & first,
    Tensor2 const & second,
    BinaryOp && operation
)
{
    static_assert( nn::is_tensor_v< std::decay_t< Tensor1 > > );
    static_assert( Tensor1::Shape::dimensions() == 2 && Tensor2::Shape::dimensions() == 1 );
    static_assert( Tensor1::Shape::size( 1 ) == Tensor2::Shape::size( 0 ) );

    constexpr TensorSize rowNumber = Tensor1::Shape::size( 0 );
    constexpr TensorSize columnNumber = Tensor1::Shape::size( 1 );
    auto const applyWithRowBroadcast = [ & ]( auto const & firstTensor, auto const & secondTensor, auto & resultTensor )
    {
        std::size_t offset = 0;
        for ( TensorSize i{ 0 }; i < rowNumber; ++i )
        {
            ranges::move
            (
                ranges::views::zip_with
                (
                    operation,
                    firstTensor.getAllElementsView().subspan( offset, columnNumber ),
                    secondTensor.getAllElementsView()
                ),
                resultTensor.getAllElementsView().subspan( offset, columnNumber ).data()
            );
            offset += columnNumber;
        }
    };

    if constexpr ( inPlace )
    {
        applyWithRowBroadcast( first, second, first );
    }
    else
    {
        using ElementType = typename Tensor1::ElementType;
        nn::Tensor< ElementType, nn::Shape< rowNumber, columnNumber > > result;
        applyWithRowBroadcast( first, second, result );
        return result;
    }
}

template
<
    bool inPlace = false,
    typename Tensor,
    typename T,
    typename BinaryOp,
    typename = std::enable_if_t< std::is_convertible_v< T, typename Tensor::ElementType > >
>
constexpr auto applyElementwiseOperation( Tensor & tensor, T const scalar, BinaryOp && operation )
{
    auto const applyOperationWithScalar = [ & ]( auto const & tensor, auto & result )
    {
        ranges::move
        (
            tensor.getAllElementsView() | ranges::views::transform
            (
                [ & ]( auto const element)
                {
                    return operation( element, scalar );
                }
            ),
            result.getAllElementsView().data()
        );
    };

    if constexpr ( inPlace )
    {
        applyOperationWithScalar( tensor, tensor );
    }
    else
    {
        nn::Tensor< typename Tensor::ElementType, typename Tensor::Shape > result;
        applyOperationWithScalar( tensor, result );
        return result;
    }
}

template< typename FirstOperand, typename SecondOperand >
struct validElementWiseOperandsHelper : std::false_type{};

template< typename ElementType, typename Shape1, TensorType type1, typename Shape2, TensorType type2 >
struct validElementWiseOperandsHelper< nn::Tensor< ElementType, Shape1, type1 >, nn::Tensor< ElementType, Shape2, type2 > > : std::true_type{};

template< typename ElementType, typename Shape, TensorType type, typename T >
struct validElementWiseOperandsHelper< nn::Tensor< ElementType, Shape, type >, T > : std::true_type
{
    static_assert( std::is_convertible_v< T, ElementType > );
};

template< typename FirstOperand, typename SecondOperand >
inline constexpr bool validElementWiseOperands = validElementWiseOperandsHelper< FirstOperand, SecondOperand >::value;

} // namespace detail

template< bool inPlace = false, typename Tensor1, typename Tensor2, typename = std::enable_if_t< !inPlace > >
constexpr auto add( Tensor1 const & first, Tensor2 const & second )
{
    static_assert( detail::validElementWiseOperands< Tensor1, Tensor2 > );
    return detail::applyElementwiseOperation< false >( first, second, std::plus< typename Tensor1::ElementType >{} );
}

template< bool inPlace = false, typename Tensor1, typename Tensor2, typename = std::enable_if_t< inPlace > >
constexpr auto add( Tensor1 & first, Tensor2 const & second )
{
    static_assert( !std::is_const_v< Tensor1 > );
    static_assert( detail::validElementWiseOperands< Tensor1, Tensor2 > );
    detail::applyElementwiseOperation< true >( first, second, std::plus< typename Tensor1::ElementType >{} );
}

template< bool inPlace = false, typename Tensor1, typename Tensor2, typename = std::enable_if_t< !inPlace > >
constexpr auto subtract( Tensor1 const & first, Tensor2 const & second )
{
    static_assert( detail::validElementWiseOperands< Tensor1, Tensor2 > );
    return detail::applyElementwiseOperation< false >( first, second, std::minus< typename Tensor1::ElementType >{} );
}

template< bool inPlace = false, typename Tensor1, typename Tensor2, typename = std::enable_if_t< inPlace > >
constexpr auto subtract( Tensor1 & first, Tensor2 const & second )
{
    static_assert( !std::is_const_v< Tensor1 > );
    static_assert( detail::validElementWiseOperands< Tensor1, Tensor2 > );
    detail::applyElementwiseOperation< true >( first, second, std::minus< typename Tensor1::ElementType >{} );
}

template< bool inPlace = false, typename Tensor1, typename Tensor2, typename = std::enable_if_t< !inPlace > >
constexpr auto multiply( Tensor1 const & first, Tensor2 const & second )
{
    static_assert( detail::validElementWiseOperands< Tensor1, Tensor2 > );
    return detail::applyElementwiseOperation< false >( first, second, std::multiplies< typename Tensor1::ElementType >{} );
}

template< bool inPlace = false, typename Tensor1, typename Tensor2, typename = std::enable_if_t< inPlace > >
constexpr auto multiply( Tensor1 & first, Tensor2 const & second )
{
    static_assert( !std::is_const_v< Tensor1 > );
    static_assert( detail::validElementWiseOperands< Tensor1, Tensor2 > );
    detail::applyElementwiseOperation< true >( first, second, std::multiplies< typename Tensor1::ElementType >{} );
}

template
<
    typename ElementType,
    typename FirstTensorShape,
    TensorType FirstTensorType,
    typename SecondTensorShape,
    TensorType SecondTensorType
>
constexpr auto dotMultiply
(
    nn::Tensor< ElementType, FirstTensorShape, FirstTensorType >   const & firstTensor,
    nn::Tensor< ElementType, SecondTensorShape, SecondTensorType > const & secondTensor
)
{
    static_assert( FirstTensorShape::dimensions() == 2, "First tensor does not have 2 dimensions" );
    static_assert( SecondTensorShape::dimensions() == 2, "Second tensor does not have 2 dimensions" );

    static_assert( FirstTensorShape::size( 1 ) == SecondTensorShape::size( 0 ), "Wrong dimensions for multiplication" );

    constexpr TensorSize commonSize = FirstTensorShape::size( 1 );

    constexpr TensorSize resultRowNum    = FirstTensorShape::size( 0 );
    constexpr TensorSize resultColumnNum = SecondTensorShape::size( 1 );
    using OutputShape = nn::Shape< resultRowNum, resultColumnNum >;

    nn::Tensor< ElementType, OutputShape > result;

    for ( TensorSize i = 0; i < resultRowNum; ++i )
    {
        for ( TensorSize j = 0; j < resultColumnNum; ++j )
        {
            for ( TensorSize k = 0; k < commonSize; ++k )
            {
                result[ i ][ j ] += firstTensor[ i ][ k ] * secondTensor[ k ][ j ];
            }
        }
    }

    return result;
}

template
<
    typename ElementType,
    TensorSize FirstTensorSize,
    TensorType FirstTensorType,
    typename SecondTensorShape,
    TensorType SecondTensorType
>
constexpr auto dotMultiply
(
    nn::Tensor< ElementType, nn::Shape< FirstTensorSize >, FirstTensorType > const & firstTensor,
    nn::Tensor< ElementType, SecondTensorShape, SecondTensorType > const & secondTensor
)
{
    static_assert( SecondTensorShape::dimensions() == 2, "Second tensor does not have 2 dimensions" );

    static_assert( FirstTensorSize == SecondTensorShape::size( 0 ), "Wrong dimensions for multiplication" );

    constexpr TensorSize commonSize = FirstTensorSize;
    constexpr TensorSize resultColumnNum = SecondTensorShape::size( 1 );

    using OutputShape = nn::Shape< resultColumnNum >;

    nn::Tensor< ElementType, OutputShape > result;

    for ( TensorSize i = 0; i < resultColumnNum; ++i )
    {
        for ( TensorSize j = 0; j < commonSize; ++j )
        {
            result[ i ] += firstTensor[ j ] * secondTensor[ j ][ i ];
        }
    }

    return result;
}

template
<
    typename ElementType,
    typename FirstTensorShape,
    TensorType FirstTensorType,
    TensorSize SecondTensorSize,
    TensorType SecondTensorType
>
constexpr auto dotMultiply
(
    nn::Tensor< ElementType, FirstTensorShape, FirstTensorType > const & firstTensor,
    nn::Tensor< ElementType, nn::Shape< SecondTensorSize >, SecondTensorType > const & secondTensor
)
{
    static_assert( FirstTensorShape::dimensions() == 2, "First tensor does not have 2 dimensions" );

    static_assert( SecondTensorSize == FirstTensorShape::size( 1 ), "Wrong dimensions for multiplication" );

    constexpr TensorSize commonSize = SecondTensorSize;
    constexpr TensorSize resultRowNum = FirstTensorShape::size( 0 );

    using OutputShape = nn::Shape< resultRowNum >;

    nn::Tensor< ElementType, OutputShape > result;

    for ( TensorSize i = 0; i < resultRowNum; ++i )
    {
        for ( TensorSize j = 0; j < commonSize; ++j )
        {
            result[ i ] += firstTensor[ i ][ j ] * secondTensor[ j ];
        }
    }

    return result;
}

template
<
    typename ElementType,
    TensorSize FirstTensorSize,
    TensorType FirstTensorType,
    TensorSize SecondTensorSize,
    TensorType SecondTensorType
>
constexpr auto dotMultiply
(
    nn::Tensor< ElementType, nn::Shape< FirstTensorSize >, FirstTensorType >   const & firstTensor,
    nn::Tensor< ElementType, nn::Shape< SecondTensorSize >, SecondTensorType > const & secondTensor
)
{
    static_assert( FirstTensorSize == SecondTensorSize, "Wrong dimensions for multiplication" );

    constexpr TensorSize commonSize = FirstTensorSize;

    ElementType result{ 0 };

    for ( TensorSize i = 0; i < commonSize; ++i )
    {
        result += firstTensor[ i ] * secondTensor[ i ];
    }

    return result;
}

template< typename OutputShape, typename ElementType, typename InputShape, TensorType TensorType, typename = std::enable_if_t< nn::is_shape_v< OutputShape > > >
constexpr auto reshape( nn::Tensor< ElementType, InputShape, TensorType > const & inputTensor )
{
    static_assert( nn::is_shape_v< OutputShape > );
    using CorrectedOutputShape =
        std::conditional_t
        <
            OutputShape::hasWildcard(),
            typename nn::detail::ShapeWithWildcardDeducer< InputShape, OutputShape >::shape,
            OutputShape
        >;
    return nn::Tensor< ElementType, CorrectedOutputShape, TensorType::regular >( inputTensor.getAllElementsView() );
}

} // namespace nn
