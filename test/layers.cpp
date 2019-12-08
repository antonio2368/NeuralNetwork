#include "layers/layer.hpp"
#include "layers/inputLayer.hpp"

#include "gtest/gtest.h"
#include "shape.hpp"

TEST( layerTest, inputSizes )
{
    nn::Layer< int, nn::Shape< 1, 2, 3 >, nn::Shape< 1, 2, 3 > > layer;
    constexpr auto inputShape = layer.inputShape();
    ASSERT_EQ( inputShape[ 0 ], 1 );
}

TEST( layerTest, inputLayer )
{
    nn::InputLayer< int, nn::Shape< 2, 3 > > layer;
    std::array< std::array< int, 3>, 2 > vector{ { { 1, 2, 3 }, { 1, 2, 3 } } };
    auto const& layerOutput = layer( vector );

    ASSERT_EQ( layerOutput[0][1], 2 );


}