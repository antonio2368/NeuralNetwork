#include "operations.hpp"
#include "tensor.hpp"
#include "utils/tensorUtils.hpp"

#include "gtest/gtest.h"

#include <vector>

TEST( operationsTest, dotMultiply )
{
    std::vector< std::vector< int > > firstOperandData{ { 1, 2, 3 }, { 2, 3, 4 } };
    nn::Tensor< int, nn::Shape< 2, 3 > > firstOperand{ firstOperandData };

    std::vector< std::vector< int > > secondOperandData{ { 3, 2 }, { 1, 4 }, { 5, 6 } };
    nn::Tensor< int, nn::Shape< 3, 2 > > secondOperand{ secondOperandData };

    nn::Tensor< int, nn::Shape< 2, 2 > > result = nn::dotMultiply( firstOperand, secondOperand );

    ASSERT_EQ( result[ 0 ][ 0 ], 20 );
    ASSERT_EQ( result[ 0 ][ 1 ], 28 );
    ASSERT_EQ( result[ 1 ][ 0 ], 29 );
    ASSERT_EQ( result[ 1 ][ 1 ], 40 );

    auto const & singleRowOperand = result[ 0 ];

    auto singleRowResult = nn::dotMultiply( result, singleRowOperand );
    ASSERT_EQ( singleRowResult[ 0 ], 1184 );
    ASSERT_EQ( singleRowResult[ 1 ], 1700 );

    singleRowResult = nn::dotMultiply( singleRowOperand, result );
    ASSERT_EQ( singleRowResult[ 0 ], 1212 );
    ASSERT_EQ( singleRowResult[ 1 ], 1680 );

    auto scalarResult = nn::dotMultiply( singleRowOperand, singleRowOperand );
    ASSERT_EQ( scalarResult, 1184 );
}

TEST( operationsTest, reshape )
{
    std::vector< std::vector< std::vector< int > > > tensorData{ { { 1, 2 }, { 2, 3 } }, { { 4, 5 }, { 6, 7 } } };
    nn::Tensor< int, nn::Shape< 2, 2, 2 > > tensor{ tensorData };

    nn::reshape< nn::Shape< 8 > >( tensor );
}