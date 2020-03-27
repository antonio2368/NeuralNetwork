#include "shape.hpp"

#include "gtest/gtest.h"

TEST( ShapeTest, dimensions )
{
    std::size_t numberOfDimensions = nn::Shape< 2, 3, 4 >::dimensions();
    ASSERT_EQ( numberOfDimensions, 3 );

    numberOfDimensions = nn::Shape<>::dimensions();
    ASSERT_EQ( numberOfDimensions, 0 );
}

TEST( ShapeTest, numberOfElements )
{
    std::size_t numberOfElements = nn::Shape< 2, 3, 4 >::numberOfElements();
    ASSERT_EQ( numberOfElements, 24 );

    numberOfElements = nn::Shape<>::numberOfElements();
    ASSERT_EQ( numberOfElements, 1 );
}