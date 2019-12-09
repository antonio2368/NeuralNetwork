#include "layer.hpp"
#include "tensor.hpp"

namespace nn
{

template< typename T, typename InputShape, int OutputNumber >
class DenseLayer : public nn::Layer< T, InputShape, Shape< InputShape::getSize( 0 ), OutputNumber > >
{
    using OutputShape = Shape< InputShape::getSize( 0 ), OutputNumber >;

    static_assert( InputShape::dimensions() == 2, "Dense layer inputs can only have 2 dimensions" );
private:
    Tensor< T, Shape< InputShape::getSize( 1 ), OutputNumber > > weights_;
public:
    constexpr auto const& getWeights() const noexcept
    {
        return weights_;
    }

    constexpr Tensor< T, OutputShape > operator()( Tensor< T, InputShape > const& ) const noexcept
    {
        return Tensor< T, OutputShape >{};
    }
};

} // namespace nn
