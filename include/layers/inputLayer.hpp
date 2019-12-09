#pragma once

#include "layer.hpp"
#include "tensor.hpp"
#include "typeTraits.hpp"

namespace nn
{

template< typename T, typename InputShape >
class InputLayer : public Layer< T, InputShape, InputShape >
{
    static_assert( InputShape::dimensions() == 4, "Input shape can only have 4 dimensions" );
private:
    Tensor< T, InputShape > input_;
public:
    template< typename Container>
    auto const& operator()( Container const& input )
    {
        input_ = Tensor< T, InputShape >{ input };
        return input_;
    }
};

} // namespace nn
