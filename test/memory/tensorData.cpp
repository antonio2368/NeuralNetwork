#include "memory/tensorData.hpp"

#include <iostream>

int main()
{
    math::memory::TensorData< double, 2, 3, 4, 6 > test;

    std::cout << "Test size: " << test.size() << std::endl;
    std::cout << "Test dimension number: " << test.dimensionNum() << std::endl;
    std::cout << "Test member size: " << test[ 0 ][ 2 ].size() << std::endl;

    math::memory::TensorData< int, 3 > vector;
    std::cout << "Vector size: " << vector.size() << std::endl;
    std::cout << "vector member: " << vector[ 0 ] << std::endl;

}