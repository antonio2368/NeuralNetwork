#include "operations.hpp"
#include "tensor.hpp"
#include "utils/tensorUtils.hpp"

#include "gtest/gtest.h"

#include <vector>

TEST( operationsTest, dotMultiply )
{
    std::vector< std::vector< int > > firstData{ { 1, 2, 3 }, { 2, 3, 4 } };
    nn::Tensor< int, nn::Shape< 2, 3 > > firstOperand{ firstData };

    std::vector< std::vector< int > > secondData{ { 3, 2 }, { 1, 4 }, { 5, 6 } };
    nn::Tensor< int, nn::Shape< 3, 2 > > secondOperand{ secondData };

    nn::Tensor< int, nn::Shape< 2, 2 > > result =  nn::dotMultiply( firstOperand, secondOperand );

    ASSERT_EQ( result[ 0 ][ 0 ].get(), 20 );
    ASSERT_EQ( result[ 0 ][ 1 ].get(), 28 );

    nn::printTensor( firstOperand );
    std::cout << '\n';
    nn::printTensor( secondOperand );
    std::cout << '\n';
    nn::printTensor( result );
    std::cout << '\n';

    std::vector< int > singleRowData{ 1, 2, 3 };
    nn::Tensor< int, nn::Shape< 3 > > singleRowTensor{ singleRowData };
    nn::Tensor< int, nn::Shape< 2 > > anotherResult = nn::dotMultiply( singleRowTensor, secondOperand );

    nn::printTensor( anotherResult );

}