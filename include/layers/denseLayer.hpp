#pragma once

#include "../tensor.hpp"

#include "../operations.hpp"

#include "../initializers/initializer.hpp"
#include "../initializers/zeroInitializer.hpp"

#include "../activations/activation.hpp"
#include "../activations/relu.hpp"

namespace nn::layer
{

namespace detail
{

template< bool hasBias, typename ElementType, std::size_t outputNumber >
struct DenseLayerBias
{
};

template< typename ElementType, std::size_t outputNumber >
struct DenseLayerBias< true, ElementType, outputNumber >
{
protected:
    using BiasTensor = Tensor< ElementType, Shape< outputNumber > >;

    template< typename T >
    inline static constexpr bool isValidBiasInitializer = std::is_same_v< std::decay_t< T >, BiasTensor > ||
                                                          initializer::is_initializer_v< std::decay_t< T > >;

    template
    <
        typename BiasInitializer = initializer::ZeroInitializer< ElementType >,
        typename = std::enable_if_t< isValidBiasInitializer< BiasInitializer > >
    >
    explicit constexpr DenseLayerBias
    (
        BiasInitializer && biasInitializer
    )
    : bias_{ std::forward< BiasInitializer >( biasInitializer ) }
    {}

    auto const & bias() const noexcept
    {
        return bias_;
    }
private:
    BiasTensor bias_;
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

    using WeightsTensor = Tensor< ElementType, Shape< InputShape::size( 1 ), outputNumber > >;

    WeightsTensor weights_;
    std::function< ElementType( ElementType const ) > activation_;

    using OutputShape = Shape< InputShape::size( 0 ), outputNumber >;

    template< typename T >
    inline static constexpr bool isValidWeightInitializer = std::is_same_v< std::decay_t< T >, WeightsTensor > ||
                                                            initializer::is_initializer_v< std::decay_t< T > >;

public:
    using OutputTensorType  = Tensor< ElementType, OutputShape >;

    static_assert( InputShape::dimensions() == 2, "Dense layer inputs can only have 2 dimension" );
    static_assert( is_tensor_v< std::decay_t< InputTensor > > );

    template
    <
        typename Activation        = activations::Relu< ElementType >,
        typename WeightInitializer = initializer::ZeroInitializer< ElementType >,
        typename BiasInitializer   = initializer::ZeroInitializer< ElementType >,
        typename = std::enable_if_t
        <
            hasBias &&
            nn::activations::is_activation_v< Activation > &&
            isValidWeightInitializer< WeightInitializer >
        >
    >
    explicit constexpr DenseLayer
    (
        Activation && activation,
        WeightInitializer && weightInitializer = initializer::ZeroInitializer< ElementType >{},
        BiasInitializer && biasInitializer = initializer::ZeroInitializer< ElementType >{}
    )
        : BaseBias{ std::forward< BiasInitializer >( biasInitializer ) },
          weights_{ std::forward< WeightInitializer >( weightInitializer ) },
          activation_{ std::forward< Activation >( activation ) }
    {}

    template
    <
        typename Activation        = activations::Relu< ElementType >,
        typename WeightInitializer = initializer::ZeroInitializer< ElementType >,
        typename = std::enable_if_t
        <
            !hasBias &&
            nn::activations::is_activation_v< Activation > &&
            isValidWeightInitializer< WeightInitializer >
        >
    >
    explicit constexpr DenseLayer
    (
        Activation && activation,
        WeightInitializer && weightInitializer = initializer::ZeroInitializer< ElementType >{}
    )
        : weights_{ std::forward< WeightInitializer >( weightInitializer ) },
          activation_{ std::forward< Activation >( activation ) }
    {}

    template< TensorType type >
    constexpr OutputTensorType operator()( Tensor< ElementType, InputShape, type > const& input ) const noexcept
    {
        auto result = nn::dotMultiply( input, weights_ );
        if constexpr ( hasBias )
        {
            nn::add< true >( result, BaseBias::bias() );
        }

        for ( auto & element : result.getAllElementsView() )
        {
            element = activation_( element );
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
