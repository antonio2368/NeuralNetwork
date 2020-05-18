#include "layers/layer.hpp"
#include "layers/inputLayer.hpp"
#include "layers/denseLayer.hpp"

#include "tensor.hpp"

#include "gtest/gtest.h"
#include "shape.hpp"

// TEST( layerTest, inputLayer )
// {
//     nn::Tensor< float, nn::Shape< 1, 1, 2 > input{ 1.0f, 2.0f };
//     nn::InputLayer< int, nn::Shape< 1, 1, 2 > > layer;

//     auto const& layerOutput = layer( vector );

//     ASSERT_EQ( layerOutput[0][0][0][0], 1 );
//     ASSERT_EQ( layerOutput[0][0][0][1], 2 );
// }

// TEST( layerTest, denseLayer )
// {
//     nn::DenseLayer< int, nn::Shape< 1, 20 >, 30 > layer;
//     auto const& outputShape = layer.outputShape();

//     ASSERT_EQ( outputShape[0], 1 );
//     ASSERT_EQ( outputShape[1], 30 );

//     nn::Tensor< int, nn::Shape< 1, 2 > > inputTensor;
//     inputTensor[0][0] = 2;
// }