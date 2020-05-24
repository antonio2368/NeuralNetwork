#pragma once

#include "tensor.hpp"

namespace nn
{

namespace layer
{

template< typename ElementType, typename InputShape  >
struct InputLayer
{
    using OutputTensorType = Tensor< ElementType, InputShape >;

private:
    OutputTensorType input_;

public:
    template< TensorType type >
    constexpr InputLayer( Tensor< ElementType, InputShape, type > const & input )
        : input_{ input }
    {}

    constexpr InputLayer( OutputTensorType && input )
        : input_{ std::move( input ) }
    {}

    auto const & operator()() noexcept
    {
        return input_;
    }
};

}

}