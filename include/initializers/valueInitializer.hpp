#pragma once

#include "initializer.hpp"

namespace nn::initializer
{

template< typename T >
class ValueInitializer : public InitializerBase< T >
{
public:
    explicit ValueInitializer( T const value ) : value_{ value }
    {}

    T getValue() const noexcept override
    {
        return value_;
    }
private:
    T value_;
};

} // namespace nn::initializer