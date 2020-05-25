#pragma once

#include "initializer.hpp"

namespace nn::initializer
{

template< typename T >
class ZeroInitializer
{
public:
    constexpr T getValue() const noexcept
    {
        return 0;
    }
};

template< typename T >
struct is_initializer< ZeroInitializer< T > > : std::true_type{};

} // namespace nn::initializer