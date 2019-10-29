#pragma once

#include "layer.hpp"

template< typename T, int... OUTPUT_SIZES >
class InputLayer : public Layer< T, OUTPUT_SIZES >
{

};