#include "tensor.hpp"

#include <type_traits>
#include <iostream>

namespace nn
{
    template< typename FirstTensor, typename SecondTensor >
    auto dotMultiply( FirstTensor const & firstTensor, SecondTensor const & secondTensor)
    {
        static_assert( is_tensor_v< FirstTensor >, "First argument is not tensor!" );
        static_assert( is_tensor_v< SecondTensor >, "Second argument is not tensor!" );

        static_assert( FirstTensor::dimensions() == 2, "First tensor does not have 2 dimensions" );
        static_assert( SecondTensor::dimensions() == 2, "Second tensor does not have 2 dimensions" );

        using FirstTensorShape = typename FirstTensor::TensorShape;
        using SecondTensorShape = typename SecondTensor::TensorShape;

        static_assert( FirstTensorShape::size( 1 ) == SecondTensorShape::size( 0 ), "Wrong dimensions for multiplication" );

        constexpr int commonSize = FirstTensorShape::size( 1 );

        using CommonType = std::common_type_t< typename FirstTensor::ElementType, typename SecondTensor::ElementType >;

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
        int FirstTensorSize,
        typename SecondTensor
    >
    auto dotMultiply( nn::Tensor< FirstTensorType, nn::Shape< FirstTensorSize > > const & firstTensor, SecondTensor const & secondTensor)
    {
        static_assert( is_tensor_v< SecondTensor >, "Second argument is not tensor!" );
        static_assert( SecondTensor::dimensions() == 2, "Second tensor does not have 2 dimensions" );

        using SecondTensorShape = typename SecondTensor::TensorShape;

        static_assert( FirstTensorSize == SecondTensorShape::size( 0 ), "Wrong dimensions for multiplication" );

        constexpr int commonSize = FirstTensorSize;
        constexpr int resultColumnNum = SecondTensorShape::size( 1 );

        using CommonType = std::common_type_t< FirstTensorType, typename SecondTensor::ElementType >;
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


} // namespace nn
