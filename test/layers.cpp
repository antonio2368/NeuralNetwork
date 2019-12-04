#include "layers/layer.hpp"
#include "layers/inputLayer.hpp"

#include "gtest/gtest.h"

TEST( layerTest, outputSizes )
{
    nn::Layer< int, 1, 2, 3 > layer;
    constexpr auto outputSizes = layer.outputSize();
    ASSERT_EQ( outputSizes[ 0 ], 1 );
}

TEST( layerTest, inputLayer )
{
    nn::InputLayer< int, 2, 3 > layer;
    std::array< std::array< int, 3>, 2 > vector{ { { 1, 2, 3 }, { 1, 2, 3 } } };
    auto const& layerOutput = layer( vector );

    std::cout << layerOutput[0][1] << '\n';
}