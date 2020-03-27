#include "utils/tensorUtils.hpp"

#include "gtest/gtest.h"

#include <vector>

TEST( utilsTest, printTensor )
{
    std::vector< std::vector< std::vector< int > > > data
    {
        { { 1, 2 }, { 2, 3 } },
        { { 4, 5 }, { 6, 7 } }
    };
    nn::Tensor< int, nn::Shape< 2, 2, 2 > > tensor{ data };

    nn::utils::printTensor( tensor );
}