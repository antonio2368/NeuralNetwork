#pragma once

#include "../tensor.hpp"

#include "../operations.hpp"

#include "../initializers/initializer.hpp"
#include "../initializers/zeroInitializer.hpp"

namespace nn::layer
{

namespace detail
{

template< bool hasBias, typename ElementType, std::size_t outputNumber >
struct DenseLayerBias
{
    template< template< typename > class Initializer >
    explicit constexpr DenseLayerBias( Initializer< ElementType > const &, initializer::enableIfInitializer< Initializer< ElementType > > * = nullptr )
    {}
};

template< typename ElementType, std::size_t outputNumber >
struct DenseLayerBias< true, ElementType, outputNumber >
{
protected:
    template< template< typename > class Initializer >
    explicit constexpr DenseLayerBias
    (
        Initializer< ElementType > const & biasInitializer = initializer::ZeroInitializer< ElementType >{},
        nn::initializer::enableIfInitializer< Initializer< ElementType > > * = nullptr
    )
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
    static_assert( is_tensor_v< std::decay_t< InputTensor > > );

    template
    <
        template< typename = ElementType > class WeightInitializer = initializer::ZeroInitializer,
        template< typename = ElementType > class BiasInitializer   = initializer::ZeroInitializer
    >
    explicit constexpr DenseLayer
    (
        WeightInitializer< ElementType > const & weightInitializer = initializer::ZeroInitializer< ElementType >{},
        BiasInitializer< ElementType > const & biasInitializer = initializer::ZeroInitializer< ElementType >{}
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

} // namespace nn::layer
