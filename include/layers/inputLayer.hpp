#pragma once

#include "layer.hpp"

namespace nn
{

template< typename T, int... OUTPUT_SIZES >
class InputLayer : public Layer< T, OUTPUT_SIZES... >
{
private:
    Tensor< T, OUTPUT_SIZES... > input_;

public:
    template< typename Container>
    auto const& operator()( Container const& input )
    {
        input_ = Tensor< T, OUTPUT_SIZES... >{ input };
        return input_;
    }
};

} // namespace nn
