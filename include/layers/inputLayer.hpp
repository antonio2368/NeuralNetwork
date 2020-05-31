#pragma once

#include "../tensor.hpp"

namespace nn::layer
{

template< typename ElementType, typename InputShape  >
struct InputLayer
{
    using OutputTensorType = Tensor< ElementType, InputShape >;

private:
    OutputTensorType input_;

public:
    template< TensorType type >
    explicit constexpr InputLayer( Tensor< ElementType, InputShape, type > const & input )
        : input_{ input }
    {}

    explicit constexpr InputLayer( OutputTensorType && input )
        : input_{ std::move( input ) }
    {}

    auto const & operator()() noexcept
    {
        return input_;
    }
};

} // namespace nn::layer