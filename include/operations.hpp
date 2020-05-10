#include "tensor.hpp"

#include <iostream>

#include <algorithm>
#include <numeric>
#include <type_traits>

namespace nn
{
    template
    <
        typename FirstElementType,
        typename FirstTensorShape,
        TensorType FirstTensorType,
        typename SecondElementType,
        typename SecondTensorShape,
        TensorType SecondTensorType
    >
    constexpr auto dotMultiply
    (
        nn::Tensor< FirstElementType, FirstTensorShape, FirstTensorType >   const & firstTensor,
        nn::Tensor< SecondElementType, SecondTensorShape, SecondTensorType > const & secondTensor
    )
    {
        static_assert( FirstTensorShape::dimensions() == 2, "First tensor does not have 2 dimensions" );
        static_assert( SecondTensorShape::dimensions() == 2, "Second tensor does not have 2 dimensions" );

        static_assert( FirstTensorShape::size( 1 ) == SecondTensorShape::size( 0 ), "Wrong dimensions for multiplication" );

        constexpr int commonSize = FirstTensorShape::size( 1 );

        using CommonType = std::common_type_t< FirstElementType, SecondElementType >;

        constexpr int resultRowNum    = FirstTensorShape::size( 0 );
        constexpr int resultColumnNum = SecondTensorShape::size( 1 );
        using OutputShape = nn::Shape< resultRowNum, resultColumnNum >;

        nn::Tensor< CommonType, OutputShape > result;

        for ( int i = 0; i < resultRowNum; ++i )
        {
            for ( int j = 0; j < resultColumnNum; ++j )
            {
                for ( int k = 0; k < commonSize; ++k )
                {
                    result[ i ][ j ] += firstTensor[ i ][ k ] * secondTensor[ k ][ j ];
                }
            }
        }

        return result;
    }

    template
    <
        typename FirstElementType,
        TensorSize FirstTensorSize,
        TensorType FirstTensorType,
        typename SecondElementType,
        typename SecondTensorShape,
        TensorType SecondTensorType
    >
    constexpr auto dotMultiply
    (
        nn::Tensor< FirstElementType, nn::Shape< FirstTensorSize >, FirstTensorType > const & firstTensor,
        nn::Tensor< SecondElementType, SecondTensorShape, SecondTensorType > const & secondTensor
    )
    {
        static_assert( SecondTensorShape::dimensions() == 2, "Second tensor does not have 2 dimensions" );

        static_assert( FirstTensorSize == SecondTensorShape::size( 0 ), "Wrong dimensions for multiplication" );

        constexpr TensorSize commonSize = FirstTensorSize;
        constexpr TensorSize resultColumnNum = SecondTensorShape::size( 1 );

        using CommonType = std::common_type_t< FirstElementType, SecondElementType >;
        using OutputShape = nn::Shape< resultColumnNum >;

        nn::Tensor< CommonType, OutputShape > result;

        for ( int i = 0; i < resultColumnNum; ++i )
        {
            for ( int j = 0; j < commonSize; ++j )
            {
                result[ i ] += firstTensor[ j ] * secondTensor[ j ][ i ];
            }
        }

        return result;
    }

    template
    <
        typename FirstElementType,
        typename FirstTensorShape,
        TensorType FirstTensorType,
        typename SecondElementType,
        TensorSize SecondTensorSize,
        TensorType SecondTensorType
    >
    constexpr auto dotMultiply
    (
        nn::Tensor< FirstElementType, FirstTensorShape, FirstTensorType >   const & firstTensor,
        nn::Tensor< SecondElementType, nn::Shape< SecondTensorSize >, SecondTensorType > const & secondTensor
    )
    {
        static_assert( FirstTensorShape::dimensions() == 2, "First tensor does not have 2 dimensions" );

        static_assert( SecondTensorSize == FirstTensorShape::size( 1 ), "Wrong dimensions for multiplication" );

        constexpr TensorSize commonSize = SecondTensorSize;
        constexpr TensorSize resultRowNum = FirstTensorShape::size( 0 );

        using CommonType = std::common_type_t< FirstElementType, SecondElementType >;
        using OutputShape = nn::Shape< resultRowNum >;

        nn::Tensor< CommonType, OutputShape > result;

        for ( int i = 0; i < resultRowNum; ++i )
        {
            for ( int j = 0; j < commonSize; ++j )
            {
                result[ i ] += firstTensor[ i ][ j ] * secondTensor[ j ];
            }
        }

        return result;
    }

    template
    <
        typename FirstElementType,
        TensorSize FirstTensorSize,
        TensorType FirstTensorType,
        typename SecondElementType,
        TensorSize SecondTensorSize,
        TensorType SecondTensorType
    >
    constexpr auto dotMultiply
    (
        nn::Tensor< FirstElementType,  nn::Shape< FirstTensorSize >, FirstTensorType >   const & firstTensor,
        nn::Tensor< SecondElementType, nn::Shape< SecondTensorSize >, SecondTensorType > const & secondTensor
    )
    {
        static_assert( FirstTensorSize == SecondTensorSize, "Wrong dimensions for multiplication" );

        constexpr TensorSize commonSize = FirstTensorSize;

        using CommonType = std::common_type_t< FirstElementType, SecondElementType >;

        CommonType result{ 0 };

        for ( int i = 0; i < commonSize; ++i )
        {
            result += firstTensor[ i ] * secondTensor[ i ];
        }

        return result;
    }

    template< typename OutputShape, typename ElementType, typename InputShape, TensorType TensorType >
    constexpr
    std::enable_if_t< nn::is_shape_v< OutputShape >, nn::Tensor< ElementType, OutputShape, TensorType::regular > >
    reshape( nn::Tensor< ElementType, InputShape, TensorType > const & inputTensor )
    {
        static_assert( InputShape::numberOfElements() == OutputShape::numberOfElements(), "Different number of elements in input tensor and expected number of elements in output tensor" );

        return nn::Tensor< ElementType, OutputShape, TensorType::regular >( inputTensor.getAllElementsView() );
    }
} // namespace nn
