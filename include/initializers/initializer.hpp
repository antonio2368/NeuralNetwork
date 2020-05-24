#pragma once

#include <type_traits>

namespace nn::initializer
{

template< typename T >
class InitializerBase
{
    static_assert( std::is_arithmetic_v< T >, "Initializer only works for arithemtic types" );

public:
    virtual T getValue() const noexcept = 0;
};

} // namespace nn::initializer