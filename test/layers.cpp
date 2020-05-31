#include <layers/denseLayer.hpp>
#include <layers/inputLayer.hpp>
#include <layers/layerUtils.hpp>

#include <initializers/valueInitializer.hpp>

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
        nn::initializer::ValueInitializer< int >{ 1 },
        nn::initializer::ValueInitializer< int >{ 2 }
    );
    auto const denseOutput = dense( inputLayer() );
    ASSERT_EQ( denseOutput[ 0 ][ 25], 5  );
    ASSERT_EQ( denseOutput[ 1 ][ 25], 14 );
}