#include "activations/relu.hpp"

#include <gtest/gtest.h>

TEST( activationTest, relu )
{
    nn::activations::Relu< double > relu;

    ASSERT_EQ( relu( 0.0  ), 0.0 );
    ASSERT_EQ( relu( -2.0 ), 0.0 );
    ASSERT_EQ( relu( 4.0  ), 4.0 );

}