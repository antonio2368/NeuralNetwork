#include "tensor.hpp"

#include <type_traits>

namespace nn
{
    template< typename FirstTensor, typename SecondTensor >
    constexpr auto dotMultiply( FirstTensor firstTensor, SecondTensor secondTensor)
    {
        static_assert( is_tensor_v< FirstTensor >, "First argument is not tensor!" );
        static_assert( is_tensor_v< SecondTensor >, "Second argument is not tensor!" );

        static_assert( FirstTensor::dimensions() == 2, "First tensor does not have 2 dimensions" );
        static_assert( SecondTensor::dimensions() == 2, "Second tensor does not have 2 dimensions" );

        using firstTensorShape = typename FirstTensor::TensorShape;
        using secondTensorShape = typename SecondTensor::TensorShape;

        static_assert( firstTensorShape::size( 1 ) == secondTensorShape::size( 0 ), "Wrong dimensions for multiplication" );

        constexpr int commonSize = firstTensorShape::size( 1 );

        using commonType = std::common_type_t< typename FirstTensor::ElementType, typename SecondTensor::ElementType >;

        constexpr int resultRowNum    = firstTensorShape::size( 0 );
        constexpr int resultColumnNum = secondTensorShape::size( 1 );
        using outputShape = nn::Shape< resultRowNum, resultColumnNum >;

        std::array< std::array< commonType, resultRowNum >, resultColumnNum > result{ { { 0 } } };

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

        return nn::Tensor< commonType, outputShape >{ result };
    }

} // namespace nn
