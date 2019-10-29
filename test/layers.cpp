#include "layers/layer.hpp"

#include "gtest/gtest.h"

TEST( layerTest, outputSizes )
{
    nn::Layer< int, 1, 2, 3 > layer;
    constexpr auto outputSizes = layer.outputSize();
    ASSERT_EQ( outputSizes[ 0 ], 1 );
}