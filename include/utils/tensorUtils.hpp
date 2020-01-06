#include "tensor.hpp"

#include <iostream>

namespace nn
{

template< typename Tensor >
void printTensor( Tensor const & tensor )
{
    static_assert( nn::is_tensor_v< Tensor >, "Can only print tensor!" );

      if constexpr ( Tensor::dimensions() == 2 )
    {
        std::cout << "[\n";
    }
    else
    {
        std::cout << '[';
    }
    for ( std::size_t i = 0; i < tensor.size(); ++i )
    {
        printTensor( tensor[ i ]);

        if constexpr ( Tensor::dimensions() == 1 )
        {
            if ( i != tensor.size() - 1 )
            {
                std::cout << ' ';
            }
        }
    }

    if constexpr ( Tensor::dimensions() == 1 )
    {
        std::cout << "]\n";
    }
    else
    {
        std::cout << ']';
    }
}

template< typename T >
void printTensor( nn::Scalar< T > const & scalar )
{
    std::cout << scalar.get();
}

} //namespace nn
