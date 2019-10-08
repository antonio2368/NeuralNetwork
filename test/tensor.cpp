#include "tensor.hpp"
#include "gtest/gtest.h"

TEST( tensorTest, dimensionNum )
{
    math::Tensor< int, 2, 1, 3 > tensorTest;
    ASSERT_EQ( tensorTest.dimensionNum(), 3 );

    math::Scalar< int > scalarTest;
    ASSERT_EQ( scalarTest.dimensionNum(), 0 );
}

TEST( scalarTest, get )
{
    math::Scalar< int > scalarTest;
    ASSERT_EQ( scalarTest.get(), 0 );
}

TEST( tensorTest, indexOperator )
{
    math::Tensor< int, 1, 3 > tensorTest;

    auto& subTensor = tensorTest[ 0 ];
    ASSERT_EQ( typeid( subTensor ), typeid( math::Tensor< int, 3 > ) );
    
    auto& scalar = subTensor [ 0 ];
    ASSERT_EQ( typeid( scalar ), typeid( math::Scalar< int > ) );
}

TEST( tensorTest, size )
{
    math::Tensor< double, 2, 1, 3 > test;
    ASSERT_EQ( test.size(), 2 );

    ASSERT_EQ( test[ 0 ].size(), 1 );
}