#include "matrix.hpp"

#include <iostream>
#include <vector>

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

    auto const& testMatrixCref = testMatrix;

    Matrix< 2, 3 > transpose = testMatrix.getTranspose();
    Matrix< 1, 2 > subMatrix = testMatrix.getSubMatrix< 1, 2 >( Coordinate{ 1, 0 } );
    Matrix< 3, 2 > copyMatrix{ testMatrix };
    auto assignMatrix = testMatrix;
    Matrix< 1, 1 > tempMatrix{ data };
    Matrix< 1, 1 > moveMatrix{ std::move( tempMatrix ) };
    auto assignMove = Matrix< 1, 1 >( data );

    std::cout << "Row num: " << testMatrix.getRowNum() << '\n';
    std::cout << "Col num: " << testMatrix.getColNum() << '\n';
    std::cout << "Max elem: " << testMatrix.max() << '\n';
    std::cout << "First row: " << testMatrix.getRow( 0 ) << '\n';
    std::cout << "Second column: " << testMatrix.getColumn( 1 ) << '\n';
    std::cout << "Matrix:\n" << testMatrix << '\n';
    std::cout << "Transpose\n" << transpose << '\n';
    std::cout << "Matrix + Matrix:\n" << testMatrix + testMatrix << '\n';
    std::cout << "Matrix - Matrix - Matrix:\n" << testMatrix - testMatrix - testMatrix << '\n';
    std::cout << "Matrix * Matrix.T:\n" << testMatrix * transpose << '\n';
    std::cout << "2 * Matrix * 4:\n" << 2 * testMatrix * 4 << '\n';
    std::cout << "submatrix:\n" << subMatrix << '\n';
    std::cout << "copy matrix:\n" << copyMatrix << '\n';
    std::cout << "assign matrix:\n" << assignMatrix << '\n';
    std::cout << "move matrix:\n" << moveMatrix << '\n';
    std::cout << "Assign move:\n" << assignMove << '\n';
}