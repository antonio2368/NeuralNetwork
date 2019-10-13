#pragma once

#include "initializer.hpp"

namespace nn
{

namespace initializer
{

template< typename T >
class ZeroInitializer : public InitializerBase< T >
{
public:
    T getValue() const noexcept override
    {
        return 0;
    }
};

} // namespace initializer

} // namespace nn