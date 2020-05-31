#include <operations.hpp>
#include <tensor.hpp>

#include <gtest/gtest.h>

#include <vector>

TEST( operationsTest, dotMultiply )
{
    nn::Tensor< int, nn::Shape< 2, 3 > > firstOperand{ 1, 2, 3, 2, 3, 4 };

    nn::Tensor< int, nn::Shape< 3, 2 > > secondOperand{ 3, 2, 1, 4, 5, 6 };

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

    auto const result2 = nn::dotMultiply( firstOperand[ 0 ], secondOperand );
    ASSERT_EQ( result2[ 0 ], 20 );
    ASSERT_EQ( result2[ 1 ], 28 );
}

TEST( operationsTest, reshape )
{
    nn::Tensor< int, nn::Shape< 2, 2, 2 > > tensor{ 0, 1, 2, 3, 4, 5, 6, 7 };

    auto reshaped = nn::reshape< nn::Shape< 4, -1 > >( tensor );
    ASSERT_EQ( reshaped[ 2 ][ 1 ], tensor[ 1 ][ 0 ][ 1 ] );

    reshaped = nn::reshape< nn::Shape< 4, 2 > >( tensor );
    ASSERT_EQ( reshaped[ 2 ][ 1 ], tensor[ 1 ][ 0 ][ 1 ] );

    auto reshaped2 = nn::reshape< nn::Shape< -1 > >( tensor );
    ASSERT_EQ( reshaped2[ 5 ], tensor[ 1 ][ 0 ][ 1 ] );

    auto reshaped3 = nn::reshape< nn::Shape< -1, 4 > >( tensor );
    ASSERT_EQ( reshaped3[ 1 ][ 1 ], tensor[ 1 ][ 0 ][ 1 ] );
}

TEST( operationTest, elementwiseOperations )
{
    nn::Tensor< std::uint32_t, nn::Shape< 2, 2, 2 > > const tensor{ 0, 1, 2, 3, 4, 5, 6, 7 };

    auto const added = nn::add( tensor, tensor );
    ASSERT_EQ( added[ 1 ][ 0 ][ 1 ], 10 );

    auto added2 = nn::add( tensor[ 0 ], tensor[ 1 ] );
    ASSERT_EQ( added2[ 0 ][ 1 ], 6 );
    ASSERT_EQ( added2[ 1 ][ 1 ], 10 );

    nn::add< true >( added2, added2 );
    ASSERT_EQ( added2[ 0 ][ 1 ], 12 );
    ASSERT_EQ( added2[ 1 ][ 1 ], 20 );

    auto added3 = nn::add( tensor[ 0 ], tensor[ 0 ][ 1 ] );
    ASSERT_EQ( added3[ 0 ][ 1 ], 4 );
    ASSERT_EQ( added3[ 1 ][ 1 ], 6 );

    nn::add< true >( added3, tensor[ 0 ][ 1 ] );
    ASSERT_EQ( added3[ 0 ][ 1 ], 7 );
    ASSERT_EQ( added3[ 1 ][ 1 ], 9 );

    auto added4 = nn::add( tensor[ 0 ], 3 );
    ASSERT_EQ( added4[ 0 ][ 1 ], 4 );
    ASSERT_EQ( added4[ 1 ][ 1 ], 6 );

    nn::add< true >( added4, 3 );
    ASSERT_EQ( added4[ 0 ][ 1 ], 7 );
    ASSERT_EQ( added4[ 1 ][ 1 ], 9 );

    auto const subtract = nn::subtract( tensor[ 0 ], tensor[ 1 ] );
    ASSERT_EQ( subtract[ 0 ][ 1 ], -4 );
    ASSERT_EQ( subtract[ 1 ][ 1 ], -4 );

    auto const multiplied = nn::multiply( tensor[ 0 ], tensor[ 1 ] );
    ASSERT_EQ( multiplied[ 0 ][ 1 ], 5 );
    ASSERT_EQ( multiplied[ 1 ][ 1 ], 21 );
}