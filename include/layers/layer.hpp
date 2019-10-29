#pragma once

#include "../tensor.hpp"
#include <iostream>

namespace nn
{

template< typename T, int... OUTPUT_SIZES >
class Layer
{
protected:
    Tensor< T, OUTPUT_SIZES... > output_;

public:
    constexpr auto outputSize() const noexcept
    {
        return std::array< int, sizeof...( OUTPUT_SIZES ) >{ { OUTPUT_SIZES... } };
    }
public:
    auto const& output() const noexcept
    {
        return output_;
    }

};

} // namespace nn