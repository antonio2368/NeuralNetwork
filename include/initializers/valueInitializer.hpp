#pragma once

#include "initializer.hpp"
namespace nn::initializer
{

template< typename T >
class ValueInitializer
{
public:
    constexpr explicit ValueInitializer( T const value ) : value_{ value }
    {}

    constexpr T getValue() const noexcept
    {
        return value_;
    }
private:
    T value_;
};

template< typename T >
struct is_initializer< ValueInitializer< T > > : std::true_type{};

} // namespace nn::initializer