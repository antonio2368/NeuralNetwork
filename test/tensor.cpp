#include "tensor.hpp"
#include "gtest/gtest.h"

TEST( tensorTest, run )
{
    math::Tensor< int, 2, 1, 3 > tensorTest;

}

TEST( tensorTest, size )
{
    math::Tensor< double, 2, 1, 3 > test;
    ASSERT_EQ( test.size(), 2 );

    ASSERT_EQ( test[ 0 ].size(), 1 );

    ASSERT_EQ( test[ 0 ][ 0 ].size(), 3 );
}