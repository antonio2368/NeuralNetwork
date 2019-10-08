#pragma once

#include <array>

namespace math
{

namespace memory
{

template < typename T, int SIZE = 0 >
class TensorData
{
private:
    std::array < T, SIZE  > data_;
public:
    TensorData() : data_{ T{} }
    {}

    T const& operator[]( std::size_t ix ) const noexcept
    {
        return data_[ ix ];
    }

    T& operator[]( std::size_t ix ) noexcept
    {
        return data_[ ix ];
    }
};

template< typename T >
class TensorData< T >
{
private:
    T data_;

public:
    TensorData() : data_{ T{} }
    {}

    T get() const noexcept
    {
        return data_;
    }
};

} // namespace memory

} // namespace math