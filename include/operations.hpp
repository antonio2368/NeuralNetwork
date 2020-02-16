#include "tensor.hpp"

#include <type_traits>
#include <iostream>

namespace nn
{
    template
    <
        typename FirstTensorType,
        typename FirstTensorShape,
        typename SecondTensorType,
        typename SecondTensorShape
    >
    auto dotMultiply
    (
        nn::Tensor< FirstTensorType, FirstTensorShape >   const & firstTensor,
        nn::Tensor< SecondTensorType, SecondTensorShape > const & secondTensor
    )
    {
        static_assert( FirstTensorShape::dimensions() == 2, "First tensor does not have 2 dimensions" );
        static_assert( SecondTensorShape::dimensions() == 2, "Second tensor does not have 2 dimensions" );

        static_assert( FirstTensorShape::size( 1 ) == SecondTensorShape::size( 0 ), "Wrong dimensions for multiplication" );

        constexpr int commonSize = FirstTensorShape::size( 1 );

        using CommonType = std::common_type_t< FirstTensorType, SecondTensorType >;

        constexpr int resultRowNum    = FirstTensorShape::size( 0 );
        constexpr int resultColumnNum = SecondTensorShape::size( 1 );
        using OutputShape = nn::Shape< resultRowNum, resultColumnNum >;

        std::array< std::array< CommonType, resultRowNum >, resultColumnNum > result{ { { 0 } } };

        for ( int i = 0; i < resultRowNum; ++i )
        {
            for ( int j = 0; j < resultColumnNum; ++j )
            {
                for ( int k = 0; k < commonSize; ++k )
                {
                    result[ i ][ j ] += firstTensor[ i ][ k ].get() * secondTensor[ k ][ j ].get();
                }
            }
        }

        return nn::Tensor< CommonType, OutputShape >{ result };
    }

    template
    <
        typename FirstTensorType,
        TensorSize FirstTensorSize,
        typename SecondTensorType,
        typename SecondTensorShape
    >
    auto dotMultiply
    (
        nn::Tensor< FirstTensorType, nn::Shape< FirstTensorSize > > const & firstTensor,
        nn::Tensor< SecondTensorType, SecondTensorShape >           const & secondTensor
    )
    {
        static_assert( SecondTensorShape::dimensions() == 2, "Second tensor does not have 2 dimensions" );

        static_assert( FirstTensorSize == SecondTensorShape::size( 0 ), "Wrong dimensions for multiplication" );

        constexpr TensorSize commonSize = FirstTensorSize;
        constexpr TensorSize resultColumnNum = SecondTensorShape::size( 1 );

        using CommonType = std::common_type_t< FirstTensorType, SecondTensorType >;
        using OutputShape = nn::Shape< resultColumnNum >;

        std::array< CommonType, resultColumnNum > result{ { 0 } };

        for ( int i = 0; i < resultColumnNum; ++i )
        {
            for ( int j = 0; j < commonSize; ++j )
            {
                result[ i ] += firstTensor[ j ].get() * secondTensor[ j ][ i ].get();
            }
        }

        return nn::Tensor< CommonType, OutputShape >{ result };
    }

    template
    <
        typename FirstTensorType,
        typename FirstTensorShape,
        typename SecondTensorType,
        TensorSize SecondTensorSize
    >
    auto dotMultiply
    (
        nn::Tensor< FirstTensorType, FirstTensorShape >               const & firstTensor,
        nn::Tensor< SecondTensorType, nn::Shape< SecondTensorSize > > const & secondTensor
    )
    {
        static_assert( FirstTensorShape::dimensions() == 2, "First tensor does not have 2 dimensions" );

        static_assert( SecondTensorSize == FirstTensorShape::size( 1 ), "Wrong dimensions for multiplication" );

        constexpr TensorSize commonSize = SecondTensorSize;
        constexpr TensorSize resultRowNum = FirstTensorShape::size( 0 );

        using CommonType = std::common_type_t< FirstTensorType, SecondTensorType >;
        using OutputShape = nn::Shape< resultRowNum >;

        std::array< CommonType, resultRowNum > result{ 0 };

        for ( int i = 0; i < resultRowNum; ++i )
        {
            for ( int j = 0; j < commonSize; ++j )
            {
                result[ i ] = firstTensor[ i ][ j ].get() * secondTensor[ j ].get();
            }
        }

        return nn::Tensor< CommonType, OutputShape >{ result };
    }

    template
    <
        typename FirstTensorType,
        TensorSize FirstTensorSize,
        typename SecondTensorType,
        TensorSize SecondTensorSize
    >
    auto dotMultiply
    (
        nn::Tensor< FirstTensorType, nn::Shape< FirstTensorSize >  >  const & firstTensor,
        nn::Tensor< SecondTensorType, nn::Shape< SecondTensorSize > > const & secondTensor
    )
    {
        static_assert( FirstTensorSize == SecondTensorSize, "Wrong dimensions for multiplication" );

        constexpr TensorSize commonSize = FirstTensorSize;

        using CommonType = std::common_type_t< FirstTensorType, SecondTensorType >;

        CommonType result{ 0 };

        for ( int i = 0; i < commonSize; ++i )
        {
            result += firstTensor[ i ].get() * secondTensor[ i ].get();
        }

        return nn::Scalar< CommonType >{ result };
    }



} // namespace nn
