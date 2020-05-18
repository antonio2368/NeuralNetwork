#pragma once

#include "layer.hpp"
#include "tensor.hpp"
#include "typeTraits.hpp"

namespace nn
{

template< typename InputShape >
class InputLayer : public Layer< InputShape, InputShape >
{
    static_assert( InputShape::dimensions() == 3, "Shape of the input can only have 3 dimensions" );
private:
public:
    // template< typename ElementType, typename TensorShape, TensorType type  >
    // constexpr InputLayer( nn::Tensor< ElementType, TensorShape, type > const & input ) : input_{ input }
    // {}

    // template< typename ElementType, typename TensorShape  >
    // constexpr InputLayer( nn::Tensor< ElementType, TensorShape, TensorType::regular > && input ) : input_{ std::move( input ) }
    // {

    // }

    // template< typename Container>
    // auto const& operator()( Container const& input )
    // {
    //     input_ = Tensor< T, InputShape >{ input };
    //     return input_;
    // }
};

} // namespace nn
