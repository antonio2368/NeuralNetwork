#pragma once

#include "../tensor.hpp"
#include <iostream>

namespace nn
{

template< typename T, int... OUTPUT_SIZES >
class Layer
{
public:
    constexpr auto outputSize() const noexcept
    {
        return std::array< int, sizeof...( OUTPUT_SIZES ) >{ { OUTPUT_SIZES... } };
    }
};

} // namespace nn