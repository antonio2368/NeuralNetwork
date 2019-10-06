#pragma once

#include <array>

template < typename T, int LEAD_SIZE, int... SIZES >
class TensorData
{
private:
    std::array < TensorData< T, SIZES... >, LEAD_SIZE > data_;
public:
    TensorData< T, SIZES... > const& operator[]( std::size_t ix ) const noexcept
    {
        return data_[ ix ];
    }

    std::size_t size() const noexcept 
    {
        return LEAD_SIZE;
    }

    std::size_t dimensionNum() const noexcept
    {
        return sizeof...( SIZES ) + 1;
    }
};

template <typename T, int LEAD_SIZE>
class TensorData< T, LEAD_SIZE >
{
private:
    std::array< T, LEAD_SIZE > data_;
public:
    TensorData() : data_{ T{} }
    {}

    T operator[]( std::size_t ix ) const noexcept
    {
        return data_[ ix ];
    }

    std::size_t size() const noexcept 
    {
        return LEAD_SIZE;
    }

    std::size_t dimensionNum() const noexcept
    {
        return 1;
    }
};