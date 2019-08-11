#include <iostream>
#include <vector>
#include "matrix.hpp"

int main()
{
    std::vector< std::vector< double > > data{ { 1.0 } };
    Matrix< 3, 2 > testMatrix
    { 
        { 
            { 1.0, 3.0 },
            { 2.0, 3.2 },
            { 4.2, 3.1 }
        } 
    };

    std::cout << "Row num: " << testMatrix.getRowNum() << '\n';
    std::cout << "Col num: " << testMatrix.getColNum() << '\n';
    std::cout << "Max elem: " << testMatrix.max() << '\n';
    std::cout << "First row: " << testMatrix.getRow( 0 ) << '\n';
    std::cout << "Second column: " << testMatrix.getColumn( 1 ) << '\n';
    std::cout << "Matrix:\n" << testMatrix << '\n';
}