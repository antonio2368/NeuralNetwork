#include "memory/tensorData.hpp"
#include "gtest/gtest.h"

TEST( tensorDataTest, indexOperator)
{
    math::memory::TensorData< int, 1, 2 > test;
    ASSERT_EQ( typeid( math::memory::TensorData< int, 2 > ), typeid( test[ 0 ] ) );
    ASSERT_EQ( typeid( int ), typeid( test[ 0 ][ 0 ] ) );
}

TEST( tensorDataTest, defaultInit )
{
    math::memory::TensorData< double, 1 > test;
    ASSERT_EQ( test[ 0 ], double{} );

    math::memory::TensorData< int, 1 > testInt;
    ASSERT_EQ( testInt[ 0 ], int{} );
}