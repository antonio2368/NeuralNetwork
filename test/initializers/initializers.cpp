#include "initializers/zeroInitializer.hpp"

#include <gtest/gtest.h>

TEST(initializerTest, zeroInitializer)
{
    nn::initializer::ZeroInitializer< int > initializer;
    ASSERT_EQ( initializer.getValue(), 0 );
}