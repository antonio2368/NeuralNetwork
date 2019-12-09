#include "layers/layer.hpp"
#include "layers/inputLayer.hpp"
#include "layers/denseLayer.hpp"

#include "gtest/gtest.h"
#include "shape.hpp"

TEST( layerTest, inputSizes )
{
    nn::Layer< int, nn::Shape< 1, 2, 3, 2 >, nn::Shape< 1, 2, 3, 4 > > layer;
    constexpr auto inputShape = layer.inputShape();
    ASSERT_EQ( inputShape[ 0 ], 1 );
}

TEST( layerTest, inputLayer )
{
    nn::InputLayer< int, nn::Shape< 1, 1, 1, 2 > > layer;
    std::array< std::array< std::array< std::array< int, 2 >, 1 >, 1 >, 1 > vector;
    vector[0][0][0][0] = 1;
    vector[0][0][0][1] = 2;
    auto const& layerOutput = layer( vector );

    ASSERT_EQ( layerOutput[0][0][0][0], 1 );
    ASSERT_EQ( layerOutput[0][0][0][1], 2 );
}

TEST( layerTest, denseLayer )
{
    nn::DenseLayer< int, nn::Shape< 1, 20 >, 30 > layer;
    auto const& outputShape = layer.outputShape();

    ASSERT_EQ( outputShape[0], 1 );
    ASSERT_EQ( outputShape[1], 30 );

    nn::Tensor< int, nn::Shape< 1, 2 > > inputTensor;
    inputTensor[0][0] = 2;
}