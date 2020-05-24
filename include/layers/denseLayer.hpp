#pragma once

#include "tensor.hpp"

#include "operations.hpp"

#include "initializers/initializer.hpp"
#include "initializers/zeroInitializer.hpp"

namespace nn
{

namespace layer
{

namespace detail
{

template< bool hasBias, typename ElementType, std::size_t outputNumber >
struct DenseLayerBias
{
    constexpr DenseLayerBias( nn::initializer::InitializerBase< ElementType > const & )
    {}
};

template< typename ElementType, std::size_t outputNumber >
struct DenseLayerBias< true, ElementType, outputNumber >
{
protected:
    constexpr DenseLayerBias( nn::initializer::InitializerBase< ElementType > const & biasInitializer = nn::initializer::ZeroInitializer< ElementType >{} )
    : bias_{ biasInitializer }
    {}

    auto const & bias() const noexcept
    {
        return bias_;
    }
private:
    Tensor< ElementType, Shape< outputNumber > > bias_;
};

}

template< typename InputLayerType, std::size_t outputNumber, bool hasBias = false >
struct DenseLayer : private detail::DenseLayerBias< hasBias, typename InputLayerType::OutputTensorType::ElementType, outputNumber >
{
private:
    using BaseBias = detail::DenseLayerBias< hasBias, typename InputLayerType::OutputTensorType::ElementType, outputNumber >;

    using InputTensor = typename InputLayerType::OutputTensorType;
    using InputShape  = typename InputTensor::Shape;
    using ElementType = typename InputTensor::ElementType;

    Tensor< ElementType, Shape< InputShape::size( 1 ), outputNumber > > weights_;

    using OutputShape = Shape< InputShape::size( 0 ), outputNumber >;
public:
    using OutputTensorType  = Tensor< ElementType, OutputShape >;

    static_assert( InputShape::dimensions() == 2, "Dense layer inputs can only have 2 dimension" );
    static_assert( is_tensor_v< std::remove_cv_t< std::remove_reference_t< InputTensor > > > );

    constexpr DenseLayer
    (
        nn::initializer::InitializerBase< ElementType > const & weightInitializer = nn::initializer::ZeroInitializer< ElementType >{},
        nn::initializer::InitializerBase< ElementType > const & biasInitializer = nn::initializer::ZeroInitializer< ElementType >{}
    )
        : BaseBias{ biasInitializer },
          weights_{ weightInitializer }
    {}

    template< TensorType type >
    constexpr OutputTensorType operator()( Tensor< ElementType, InputShape, type > const& input ) const noexcept
    {
        auto result = nn::dotMultiply( input, weights_ );
        if constexpr ( hasBias )
        {
            nn::add< true >( result, BaseBias::bias() );
        }
        return result;
    }
};

template< std::size_t outputNumber, bool hasBias, typename InputLayer, typename... Args >
constexpr auto createDenseLayer( InputLayer const &, Args&&... args )
{
    return DenseLayer< InputLayer, outputNumber, hasBias >{ std::forward< Args >( args )... };
}

}

} // namespace nn
