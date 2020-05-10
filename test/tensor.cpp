#include "tensor.hpp"
#include "shape.hpp"

#include "gtest/gtest.h"

TEST( tensorTest, dimensionNum )
{
    auto dimensions = nn::Tensor< int, nn::Shape< 2, 1, 3 > >::dimensions();
    ASSERT_EQ( dimensions, 3 );
}

TEST( tensorTest, initializer )
{
    nn::Tensor< int, nn::Shape< 2, 1 > > const test;
    ASSERT_EQ( test[0][0], 0 );
}

TEST( tensorTest, indexOperator )
{
    nn::Tensor< int, nn::Shape< 2, 3 > > tensorTest{ 1, 2, 3, 4, 5, 6 };

    ASSERT_EQ( tensorTest[ 0 ][ 1 ], 2 );

    nn::Tensor< int, nn::Shape< 3 > > subTensor( tensorTest[ 1 ] );
    ASSERT_EQ( subTensor[ 1 ], 5 );

    auto const & subTensor2{ tensorTest[ 1 ] };
    subTensor2[ 1 ] = 20;
    ASSERT_EQ( subTensor2[ 0 ], 4 );
    ASSERT_EQ( subTensor2[ 1 ], 20 );
    ASSERT_EQ( tensorTest[ 1 ][ 1 ], 20 );
    ASSERT_EQ( subTensor[ 1 ], 5 );

    nn::Tensor< int, nn::Shape< 2, 3 > > const tensorConst1{ 1, 2, 3, 4, 5, 6 };
    ASSERT_EQ( tensorConst1[ 0 ][ 1 ], 2 );

    nn::Tensor< int, nn::Shape< 2 > > const tensorRow1{ 1, 2 };
    ASSERT_EQ( tensorRow1[ 0 ], 1 );

    nn::Tensor< int, nn::Shape< 2 > > tensorRow2{ 1, 2 };
    tensorRow2[ 1 ] = 20;
    ASSERT_EQ( tensorRow2[ 1 ], 20 );
    ASSERT_EQ( tensorRow2[ 0 ], 1 );

    nn::Tensor< int, nn::Shape< 1, 1, 1 > > deepTensor{ 1 };
    ASSERT_EQ( deepTensor[ 0 ][ 0 ][ 0 ], 1 );
}

TEST( tensorTest, assignmentOperator )
{
    nn::Tensor< int, nn::Shape< 1, 3 > > tensorTest;

    tensorTest[ 0 ][ 1 ] = 1;

    int const number = 2;
    tensorTest[ 0 ][ 2 ] = number;

    ASSERT_EQ( tensorTest[ 0 ][ 0 ], 0 );
    ASSERT_EQ( tensorTest[ 0 ][ 2 ], number );
    ASSERT_EQ( tensorTest[ 0 ][ 1 ], 1 );
}

TEST( tensorTest, containerInitializer )
{
    std::vector< int > data{ 1, 2, 3, 4, 5, 6 };

    nn::Tensor< int, nn::Shape< 2, 3 > > const test( data );
    ASSERT_EQ( test[ 0 ][ 1 ], data[ 1 ] );
    ASSERT_EQ( test[ 1 ][ 2 ], data[ 5 ] );

    nn::Tensor< int, nn::Shape< 2, 2 > > testInitList{ 1, 2, 3, 4 };
    ASSERT_EQ( testInitList[ 0 ][ 1 ], 2 );
    ASSERT_EQ( testInitList[ 1 ][ 1 ], 4 );
}

TEST( tensorTest, copyTensor )
{
    nn::Tensor< int, nn::Shape< 2, 3 > > tensor1{ 1, 2, 3, 4, 5, 6 };
    auto tensor2{ tensor1 };

    tensor1[ 0 ][ 1 ] = 3;
    ASSERT_EQ( tensor1[ 0 ][ 1 ], 3 );
    ASSERT_EQ( tensor2[ 0 ][ 1 ], 2 );

    tensor2 = tensor1;
    ASSERT_EQ( tensor1[ 0 ][ 1 ], 3 );
    ASSERT_EQ( tensor2[ 0 ][ 1 ], 3 );

    auto const& tensorView = tensor1[ 0 ];
    tensorView[ 1 ] = 10;
    ASSERT_EQ( tensorView[ 1 ], 10 );
    ASSERT_EQ( tensor1[ 0 ][ 1], 10 );

    nn::Tensor< int, nn::Shape< 3 > > tensor3( tensorView );
    ASSERT_EQ( tensor3[ 0 ], 1 );

    auto movedTensor1 = std::move( tensor1 );
    ASSERT_EQ( movedTensor1[ 0 ][ 1 ], 10 );

    auto const movedTensor2{ nn::Tensor< int, nn::Shape< 2, 3 > >{ 1, 2, 3, 4, 5, 6 } };
    ASSERT_EQ( movedTensor2[ 0 ][ 1 ], 2 );
}