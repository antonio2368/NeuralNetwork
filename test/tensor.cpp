#include "tensor.hpp"
#include "shape.hpp"

#include "gtest/gtest.h"

TEST( tensorTest, dimensionNum )
{
    nn::Tensor< int, nn::Shape< 2, 1, 3 > > tensorTest;
    ASSERT_EQ( tensorTest.dimensions(), 3 );

    nn::Scalar< int > scalarTest;
    ASSERT_EQ( scalarTest.dimensions(), 0 );
}

TEST( tensorTest, run )
{
    nn::Tensor< int, nn::Shape< 2, 1 > > test;
    ASSERT_EQ( test[0][0], 0 );

    test[0][0] = 3;
    ASSERT_EQ( test[0][0], 3 );
}

TEST( scalarTest, get )
{
    nn::Scalar< int > scalarTest;
    ASSERT_EQ( scalarTest.get(), 0 );
    ASSERT_EQ( scalarTest, 0 );
}

TEST( tensorTest, indexOperator )
{
    nn::Tensor< int, nn::Shape< 1, 3 > > tensorTest;

    auto& subTensor = tensorTest[ 0 ];
    ASSERT_EQ( typeid( subTensor ), typeid( nn::Tensor< int, nn::Shape< 3 > > ) );

    auto& scalar = subTensor [ 0 ];
    ASSERT_EQ( typeid( scalar ), typeid( nn::Scalar< int > ) );
}

TEST( tensorTest, size )
{
    nn::Tensor< double, nn::Shape< 2, 1, 3 > > test;
    ASSERT_EQ( test.size(), 2 );

    ASSERT_EQ( test[ 0 ].size(), 1 );
}