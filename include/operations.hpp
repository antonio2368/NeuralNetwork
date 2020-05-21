#include "tensor.hpp"
#include "detail/shapeOperations.hpp"

#include <range/v3/view/zip.hpp>
#include <range/v3/algorithm/move.hpp>

#include <algorithm>
#include <numeric>
#include <type_traits>

namespace nn
{

namespace detail
{

template< typename BinaryOp, typename Tensor, typename = std::enable_if_t< nn::is_tensor_v< Tensor > > >
constexpr auto applyElementwiseOperation( Tensor const & first, Tensor const & second, BinaryOp operation )
{
    nn::Tensor< typename Tensor::ElementType, typename Tensor::Shape, TensorType::regular > result;
    ranges::move( ranges::views::zip_with(operation, first.getAllElementsView(), second.getAllElementsView() ), result.getAllElementsView().data() );

    return result;
}

} // namespace detail

template< typename Tensor, typename = std::enable_if_t< nn::is_tensor_v< Tensor > > >
constexpr auto add( Tensor const & first, Tensor const & second )
{
    return detail::applyElementwiseOperation( first, second, std::plus< typename Tensor::ElementType >{} );
}

template< typename Tensor, typename = std::enable_if_t< nn::is_tensor_v< Tensor > > >
constexpr auto subtract( Tensor const & first, Tensor const & second )
{
    return detail::applyElementwiseOperation( first, second, std::minus< typename Tensor::ElementType >{} );
}

template< typename Tensor, typename = std::enable_if_t< nn::is_tensor_v< Tensor > > >
constexpr auto multiply( Tensor const & first, Tensor const & second )
{
    return detail::applyElementwiseOperation( first, second, std::multiplies< typename Tensor::ElementType >{} );
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
    using CorrectedOutputShape =
        std::conditional_t
        <
            nn::detail::hasWildcard< OutputShape >,
            typename nn::detail::ShapeWithWildcardDeducer< InputShape, OutputShape >::shape,
            OutputShape
        >;
    return nn::Tensor< ElementType, CorrectedOutputShape, TensorType::regular >( inputTensor.getAllElementsView() );
}

} // namespace nn
