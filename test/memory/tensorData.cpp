#include "memory/tensorData.hpp"
#include "gtest/gtest.h"

TEST( tensorDataTest, indexOperator)
{
    math::memory::TensorData< int, 5 > test;
    ASSERT_EQ( typeid( int ), typeid( test[ 0 ] ) );
}

TEST( tensorDataTest, defaultInit )
{
    math::memory::TensorData< double, 1 > test;
    ASSERT_EQ( test[ 0 ], double{} );

    math::memory::TensorData< int, 1 > testInt;
    ASSERT_EQ( testInt[ 0 ], int{} );
}