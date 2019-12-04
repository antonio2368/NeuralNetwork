#include "tensor.hpp"

#include "gtest/gtest.h"

TEST( tensorTest, dimensionNum )
{
    nn::Tensor< int, 2, 1, 3 > tensorTest;
    ASSERT_EQ( tensorTest.dimensionNum(), 3 );

    nn::Scalar< int > scalarTest;
    ASSERT_EQ( scalarTest.dimensionNum(), 0 );
}

TEST( tensorTest, run )
{
    nn::Tensor< int, 2, 1 > test;
    ASSERT_EQ( test[0][0], 0 );

    test[0][0] = 3;
    ASSERT_EQ( test[0][0], 3 );
}

TEST( scalarTest, get )
{
    nn::Scalar< int > scalarTest;
    ASSERT_EQ( scalarTest.get(), 0 );
}

TEST( tensorTest, indexOperator )
{
    nn::Tensor< int, 1, 3 > tensorTest;

    auto& subTensor = tensorTest[ 0 ];
    ASSERT_EQ( typeid( subTensor ), typeid( nn::Tensor< int, 3 > ) );

    auto& scalar = subTensor [ 0 ];
    ASSERT_EQ( typeid( scalar ), typeid( nn::Scalar< int > ) );
}

TEST( tensorTest, size )
{
    nn::Tensor< double, 2, 1, 3 > test;
    ASSERT_EQ( test.size(), 2 );

    ASSERT_EQ( test[ 0 ].size(), 1 );
}

TEST( tensorTest, initializeFromContainer )
{
    std::array< std::array< int, 3>, 2 > vector{ { { 1, 2, 3 }, { 1, 2, 3 } } };
    nn::Tensor< int, 2, 3 > tensor{ vector };

    std::cout << tensor[1][1] << '\n';
}