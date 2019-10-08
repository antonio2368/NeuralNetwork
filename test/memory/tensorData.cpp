#include "memory/tensorData.hpp"
#include "gtest/gtest.h"

TEST(tensorDataTest, size)
{
    math::memory::TensorData< double, 2, 1, 3 > test;
    ASSERT_EQ( test.size(), 2 );
}