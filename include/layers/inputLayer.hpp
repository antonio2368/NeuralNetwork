#pragma once

#include "layer.hpp"

namespace nn
{

template< typename T, typename InputSize >
class InputLayer : public Layer< T, InputSize, InputSize >
{
private:
    Tensor< T, InputSize > input_;

public:
    template< typename Container>
    auto const& operator()( Container const& input )
    {
        input_ = Tensor< T, InputSize >{ input };
        return input_;
    }
};

} // namespace nn
