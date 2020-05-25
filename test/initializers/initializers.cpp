#include "initializers/zeroInitializer.hpp"
#include "initializers/valueInitializer.hpp"

#include <gtest/gtest.h>

TEST(initializerTest, zeroInitializer)
{
    nn::initializer::ZeroInitializer< int > initializer;
    ASSERT_EQ( initializer.getValue(), 0 );
}

TEST( initializerTest, valueInitializer )
{
    nn::initializer::ValueInitializer< float > initializer{ 1.f };
    ASSERT_EQ( initializer.getValue(), 1.f );
}