#pragma once

#include <type_traits>

namespace nn
{

namespace initializer
{

template< typename T >
class InitializerBase
{
    static_assert( std::is_arithmetic_v< T >, "Initializer only works for arithemtic types" );

public:
    virtual T getValue() const noexcept = 0;
};

} // namespace initializer

} // namespace nn