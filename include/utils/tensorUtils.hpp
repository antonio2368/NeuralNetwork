#include "tensor.hpp"

#include <iostream>

namespace nn
{

namespace utils
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

} //namespace utils

} //namespace nn
