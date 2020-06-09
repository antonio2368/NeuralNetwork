#include <layers/denseLayer.hpp>
#include <layers/inputLayer.hpp>
#include <layers/layerUtils.hpp>

#include <initializers/valueInitializer.hpp>
#include <activations/relu.hpp>

#include <tensor.hpp>

#include <shape.hpp>

#include <gtest/gtest.h>

#include <range/v3/view/zip.hpp>

TEST( layerTest, inputLayer )
{
    nn::Tensor< int, nn::Shape< 2, 3 > > const input{ 0, 1, 2, 3, 4, 5 };
    nn::layer::InputLayer inputLayer{ input };

    for ( auto const & [ rawTensorElement, inputLayerTensorElement ] : ranges::views::zip( input.getAllElementsView(), inputLayer().getAllElementsView() ) )
    {
        ASSERT_EQ( rawTensorElement, inputLayerTensorElement );
    }
}

TEST( layerTest, denseLayer )
{
    nn::Tensor< int, nn::Shape< 2, 3 > > input{ 0, 1, 2, 3, 4, 5 };

    nn::layer::InputLayer inputLayer{ input };
    auto const dense = nn::layer::createDenseLayer< 30, true >
    (
        inputLayer,
        nn::activations::Relu< int >{},
        nn::initializer::ValueInitializer< int >{ 1 },
        nn::Tensor< int, nn::Shape< 30 > >{}
    );
    auto const denseOutput = dense( inputLayer() );
    ASSERT_EQ( denseOutput[ 0 ][ 25], 3  );
    ASSERT_EQ( denseOutput[ 1 ][ 25], 12 );

    auto const dense2 = nn::layer::createDenseLayer< 2, false >
    (
        inputLayer,
        nn::activations::Relu< int >{},
        nn::Tensor< int, nn::Shape< 3, 2 > >{ 1, 2, 3, 4, -5, 6 }
    );

    auto const denseOutput2 = dense2( inputLayer() );
    ASSERT_EQ( denseOutput2[ 0 ][ 0 ], 0 );
    ASSERT_EQ( denseOutput2[ 0 ][ 1 ], 16 );
}