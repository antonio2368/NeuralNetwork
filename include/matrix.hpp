#ifndef MATRIX_H
#define MATRIX_H

#include "zip.hpp"
#include "coordinate.hpp"

#include <array>
#include <exception>
#include <algorithm>
#include <cassert>
#include <iostream>
#include <iomanip>

template< int N1, int N2 >
class Matrix;

namespace
{
    constexpr int doublePrecision = 5;
    constexpr int outputWidth = 9;

    constexpr bool isPowerOf2( int n ) noexcept
    {
        int oneCount = 0;
        while ( n != 0 )
        {
            oneCount += n & 1;
            n >>= 1;
        }

        return oneCount == 1;
    }
}

template< int N1, int N2 >
class Matrix
{
    static_assert( N1 > 0 && N2 > 0, "Dimension must be positive integers." );

private:
    template< int K1, int K2 >
    using MatrixData = std::array< std::array< double, K2 >, K1 >;

    MatrixData< N1, N2 > data;

    template< template< typename > class RowCollection, 
              template< typename > class ColumnCollection >
    void createMatrixFromCollection( RowCollection< ColumnCollection< double > > const&  );

    template< int K1, int K2, typename Function >
    friend constexpr Matrix< K1, K2 > matrixOperation( Matrix< K1, K2 > const&, Matrix< K1, K2 > const&, Function );

    template< std::size_t rowNum, std::size_t colNum, int K1, int K2, int K3, int K4, typename Function >
    friend constexpr Matrix< rowNum , colNum > binaryOperationSubmatrix( Matrix< K1 , K2 > const&, Matrix< K3 , K4 > const &, Coordinate const&, Coordinate const&, Function ) noexcept;

    template< int K1, int K2, int K3, int K4 >
    friend constexpr void copySubMatrix( Matrix< K1, K2 >&, Matrix< K3, K4 >&&, Coordinate const&, std::size_t, std::size_t ) noexcept;

    template< int K1, int K2, int K3 >
    friend constexpr Matrix< K1, K3 > classicMatrixMultiplication( Matrix< K1, K2 > const&, Matrix< K2, K3 > const& ) noexcept;

    template< int K >
    friend constexpr Matrix< K, K > strassanMatrixMultiplication( Matrix< K, K > const&, Matrix< K, K > const& ) noexcept;

    template< int K  >
    friend constexpr Matrix< K , K  > strassanMatrixMultiplicationRecurse( Matrix< K , K  > &&, Matrix< K , K  > && ) noexcept;
public:
    Matrix();

    Matrix( MatrixData< N1, N2 >&& data ) : data{ std::move( data ) }
    {}

    template< template< typename > class RowCollection, 
              template< typename > class ColumnCollection >
    Matrix( RowCollection< ColumnCollection< double > > const& );

    Matrix( std::initializer_list< std::initializer_list< double > >const & );

    Matrix( Matrix< N1, N2 > const & ) = default;
    
    Matrix( Matrix< N1, N2 >&& ) = default;

    constexpr std::size_t getRowNum() const noexcept
    {
        return N1;
    }

    constexpr std::size_t getColNum() const noexcept
    {
        return N2;
    }

    double max() const noexcept;

    constexpr std::array< double, N2 > getRow( std::size_t ) const noexcept;

    constexpr std::array< double, N1 > getColumn( std::size_t ) const noexcept;

    constexpr Matrix< N2, N1 > getTranspose() const noexcept;

    Matrix< N1, N2 > operator=( Matrix< N1, N2 > const& );
    
    Matrix< N1, N2 > operator=( Matrix< N1, N2 >&& );

    template< std::size_t rowNum, std::size_t colNum >
    constexpr Matrix< rowNum, colNum > getSubMatrix( Coordinate const& ) const noexcept;

    template< int K1, int K2 >
    friend constexpr Matrix< K1, K2 > operator-( Matrix< K1, K2 >, const Matrix< K1, K2 >& );

    template< int K1, int K2 >
    friend constexpr Matrix< K1, K2 > operator+( Matrix< K1, K2 >, const Matrix< K1, K2 >& );

    template< int K1, int K2, int K3 >
    friend constexpr Matrix< K1, K3 > operator*( Matrix< K1, K2 >, const Matrix< K2, K3 >& );

    template< int K1, int K2 >
    friend constexpr Matrix< K1, K2 > operator*( Matrix< K1, K2 > const&, double );

    template< int K1, int K2 >
    friend constexpr Matrix< K1, K2 > operator*( double, Matrix< K1, K2 > const& );

    template< std::size_t K >
    friend std::ostream& operator<<( std::ostream&, std::array< double, K > const& );

    template< int K1, int K2 >
    friend std::ostream& operator<<( std::ostream&, Matrix< K1, K2 > const& );

    template< int, int >
    friend class Matrix;

};

template< int N1, int N2 >
Matrix< N1, N2 > Matrix< N1, N2 >::operator=( Matrix< N1, N2 > const& other )
{
    return Matrix< N1, N2 >( other );
}

template< int N1, int N2 >
Matrix< N1, N2 > Matrix< N1, N2 >::operator=( Matrix< N1, N2 >&& other )
{
    return Matrix< N1, N2 >( std::move( other ) );
}

template< int K1, int K2 >
constexpr Matrix< K1, K2 > operator*( Matrix< K1, K2 > const& lhs, double scalar )
{
    Matrix< K1, K2 > result;

    for ( std::size_t i = 0; i < K1; ++i )
    {
        for ( std::size_t j = 0; j < K2; ++j )
        {
            result.data[ i ][ j ] = lhs.data[ i ][ j ] * scalar;
        }
    }

    return result;
}

template< int K1, int K2 >
constexpr Matrix< K1, K2 > operator*( double scalar, Matrix< K1, K2 > const& lhs )
{
    return lhs * scalar;
}

template< int K1, int K2 >
constexpr Matrix< K2, K1 > Matrix< K1, K2 >::getTranspose() const noexcept
{
    Matrix< K2, K1 > result;

    for ( std::size_t i = 0; i < K2; ++i )
    {
        auto column = getColumn( i );
        for ( std::size_t j = 0; j < K1; ++j )
        {
            result.data[ i ][ j ] = column[ j ];
        }
    }

    return result;
}

template< int K1, int K2, int K3, int K4 >
constexpr void copySubMatrix( Matrix< K1, K2 >& destination, Matrix< K3, K4 >&& source, Coordinate const& coordinate, std::size_t rowNum, std::size_t colNum ) noexcept
{
    assert( coordinate.first() < K1 && coordinate.second() < K2 );
    assert( coordinate.first() + rowNum <= K1 && coordinate.second() + colNum <= K2 );

    for ( int i1 = coordinate.first(), i2 = 0; i2 < rowNum; ++i1, ++i2 )
    {
        for ( int j1 = coordinate.second(), j2 = 0; j2 < colNum; ++j1, ++j2 )
        {
            destination.data[ i1 ][ j1 ] = source.data[ i2 ][ j2 ];
        }
    }
}

template< std::size_t rowNum, std::size_t colNum, int K1, int K2, int K3, int K4, typename Function >
constexpr Matrix< rowNum , colNum > binaryOperationSubmatrix( Matrix< K1 , K2 > const& lhs, Matrix< K3 , K4 > const & rhs, Coordinate const& firstCoordinate, Coordinate const& secondCoordinate, Function operation ) noexcept
{
    assert( firstCoordinate.first() < K1 && firstCoordinate.second() < K2 );
    assert( secondCoordinate.first() < K3 && secondCoordinate.second() < K4 );
    assert( firstCoordinate.first() + rowNum <= K1 && firstCoordinate.second() + colNum <= K2 );
    assert( secondCoordinate.first() + rowNum <= K3 && secondCoordinate.second() + colNum <= K4 );

    Matrix< rowNum, colNum > result;

    for ( int i1 = firstCoordinate.first(), i2 = secondCoordinate.first(), i3 = 0; i3 < rowNum; ++i1, ++i2, ++i3 )
    {
        for ( int j1 = firstCoordinate.second(), j2 = secondCoordinate.second(), j3 = 0; j3 < colNum; ++j1, ++j2, ++j3 )
        {
            result.data[ i3 ][ j3 ] = operation( lhs.data[ i1 ][ j1 ], rhs.data[ i2 ][ j2 ] );
        }
    }

    return result;
}

template< int N1, int N2 >
template< std::size_t rowNum, std::size_t colNum >
constexpr Matrix< rowNum, colNum > Matrix< N1, N2 >::getSubMatrix( Coordinate const& coordinate ) const noexcept
{
    assert( coordinate.first() < N1 && coordinate.second() < N2 );
    assert( coordinate.first() + rowNum <= N1 && coordinate.second() + colNum <= N2 );

    Matrix< rowNum, colNum > result;

    for ( int i1 = coordinate.first(), i2 = 0; i2 < rowNum; ++i1, ++i2 )
    {
        for ( int j1 = coordinate.second(), j2 = 0; j2 < colNum; ++j1, ++j2 )
        {
            result.data[ i2 ][ j2 ] = data[ i1 ][ j1 ];
        }
    }

    return result;
}


template< int K  >
constexpr Matrix< K , K  > strassanMatrixMultiplicationRecurse( Matrix< K , K  > && lhs, Matrix< K , K  > && rhs ) noexcept
{
    Matrix< K, K > result;

    if constexpr ( K  == 1 )
    {
        result.data[0][0] = lhs.data[0][0] * rhs.data[0][0];
        return result;
    }
    else
    {        
        constexpr std::size_t subMatrixSize = K / 2;

        Coordinate upperLeft{ 0, 0 };
        Coordinate upperRight{ subMatrixSize, 0 };
        Coordinate lowerLeft{ 0, subMatrixSize };
        Coordinate lowerRight{ subMatrixSize, subMatrixSize };

        auto addSubmatrices = [ & ]( auto const& lhs, auto const& rhs, auto const& firstCoordinate, auto const& secondCoordinate )
        {
            Matrix< subMatrixSize, subMatrixSize > result = binaryOperationSubmatrix< subMatrixSize, subMatrixSize >( lhs, rhs, firstCoordinate, secondCoordinate, []( auto a, auto b ) { return a + b; } );
            return result;
        };

        auto subtractSubmatrices = [ & ]( auto const& lhs, auto const& rhs, auto const& firstCoordinate, auto const& secondCoordinate )
        {
            Matrix< subMatrixSize, subMatrixSize > result = binaryOperationSubmatrix< subMatrixSize, subMatrixSize >( lhs, rhs, firstCoordinate, secondCoordinate, []( auto a, auto b ) { return a - b; } );
            return result;
        };

        auto M1 = strassanMatrixMultiplicationRecurse
        ( 
            addSubmatrices( lhs, lhs, upperLeft, lowerRight ),
            addSubmatrices( rhs, rhs, upperLeft, lowerRight )
        );
        auto M2 = strassanMatrixMultiplicationRecurse
        (
            addSubmatrices( lhs, lhs, lowerLeft, lowerRight ),
            rhs.template getSubMatrix< subMatrixSize, subMatrixSize >( upperLeft )
        );
        auto M3 = strassanMatrixMultiplicationRecurse
        (
            lhs.template getSubMatrix< subMatrixSize, subMatrixSize >( upperLeft ),
            subtractSubmatrices( rhs, rhs, upperRight, lowerRight )
        );
        auto M4 = strassanMatrixMultiplicationRecurse
        (
            lhs.template getSubMatrix< subMatrixSize, subMatrixSize >( lowerRight ),
            subtractSubmatrices( rhs, rhs, lowerLeft, upperLeft )
        );
        auto M5 = strassanMatrixMultiplicationRecurse
        (
            addSubmatrices( lhs, lhs, upperLeft, upperRight ),
            rhs.template getSubMatrix< subMatrixSize, subMatrixSize >( lowerRight )
        );
        auto M6 = strassanMatrixMultiplicationRecurse
        (
            subtractSubmatrices( lhs, lhs, lowerLeft, upperLeft ),
            addSubmatrices( rhs, rhs, upperLeft, upperRight )
        );
        auto M7 = strassanMatrixMultiplicationRecurse
        (
            subtractSubmatrices( lhs, lhs, upperRight, lowerRight ),
            addSubmatrices( rhs, rhs, lowerLeft, lowerRight )
        );

        copySubMatrix( result, M1 + M4 - M5 + M7, upperLeft,  subMatrixSize, subMatrixSize );
        copySubMatrix( result, M3 + M5,           upperRight, subMatrixSize, subMatrixSize );
        copySubMatrix( result, M2 + M4,           lowerLeft,  subMatrixSize, subMatrixSize );
        copySubMatrix( result, M1 - M2 + M3 + M6, lowerRight, subMatrixSize, subMatrixSize );

        return result;
    }

}

template< int K  >
constexpr Matrix< K , K  > strassanMatrixMultiplication( Matrix< K , K  > const& lhs, Matrix< K , K  > const& rhs ) noexcept
{
    static_assert( isPowerOf2( K  ), "Strassen can only be applied to square matrices with dimension that are power of 2" );

    Matrix< K, K > lhsTemp{ lhs };
    Matrix< K, K > rhsTemp{ rhs };

    return strassanMatrixMultiplicationRecurse( std::move( lhsTemp ), std::move( lhsTemp ) );
}

template< int K1, int K2, int K3 >
constexpr Matrix< K1, K3 > classicMatrixMultiplication( Matrix< K1, K2 > const& lhs, Matrix< K2, K3 > const& rhs ) noexcept
{
    Matrix< K1, K3 > result;
    for ( std::size_t i = 0; i < K1; ++i )
    {
        for  ( std::size_t j = 0; j < K3; ++ j )
        {
            auto& row = lhs.data[ i ];
            auto column = rhs.getColumn( j );

            ZipIterator zipIter{ row.begin(), row.end(), column.begin(), column.end() };

            while ( !zipIter.isEnd() )
            {
                result.data[ i ][ j ] += ( *zipIter ).first * ( *zipIter ).second;
                ++zipIter;
            }
        }
    }

    return result;

}

template< int K1, int K2, int K3 >
constexpr Matrix< K1, K3 > operator*( Matrix< K1, K2 > lhs, const Matrix< K2, K3 >& rhs )
{
    auto const& lhsRef = lhs;

    if constexpr ( K1 == K2 && K1 == K3 && isPowerOf2( K1 ) )
    {
        std::cout << "using strassan\n";
        return strassanMatrixMultiplication( lhs, rhs );
    }
    else
    {
        std::cout << "using classic\n";
        return classicMatrixMultiplication( lhsRef, rhs );
    }
}

template< int K1, int K2 >
constexpr Matrix< K1, K2 > operator-( Matrix< K1, K2 > lhs, const Matrix< K1, K2 >& rhs )
{
    return matrixOperation( lhs, rhs, []( double a, double b ) { return a - b; } );
}

template< int K1, int K2 >
constexpr Matrix< K1, K2 > operator+( Matrix< K1, K2 > lhs, const Matrix< K1, K2 >& rhs )
{
    return matrixOperation( lhs, rhs, []( double a, double b ) { return a + b; } );
}

template< int N1, int N2 >
constexpr std::array< double, N1 > Matrix< N1, N2 >::getColumn( std::size_t index ) const noexcept
{
    assert( index < N2 );

    std::array< double, N1 > result;

    std::transform( data.begin(), data.end(), result.begin(), [ &index ]( auto row )
    {
        return row[ index ];
    });

    return result;
}

template< int N1, int N2 >
constexpr std::array< double, N2 > Matrix< N1, N2 >::getRow( std::size_t index ) const noexcept
{
    assert( index < N1 );
    return data[ index ];
}

template< int N1, int N2 >
double Matrix< N1, N2 >::max() const noexcept
{
    std::array< double, N1 > maxInRows;
    std::transform
    ( 
        data.begin(),
        data.end(), 
        maxInRows.begin(), 
        []( std::array< double, N2 > row )
        { 
            return *std::max_element( row.begin(), row.end() ); 
        } 
    );
    return *std::max_element( maxInRows.begin(), maxInRows.end() );
}

template< std::size_t K >
std::ostream& operator<<( std::ostream& os, std::array< double, K > const& array )
{
    for ( auto& elem : array )
    {
       os << std::setprecision( doublePrecision ) << std::setw( outputWidth ) << elem << ' ';
    }

    return os;
}

template< int K1, int K2 >
std::ostream& operator<<( std::ostream& os, Matrix< K1, K2 > const& matrix )
{
    os << '[' << '\n';
    for ( auto& elem : matrix.data )
    {
        os << elem << '\n';
    }

    os << ']' << '\n';

    return os;
}

template< int N1, int N2 >
template< template< typename > class RowCollection, 
          template< typename > class ColumnCollection >
void Matrix< N1, N2 >::createMatrixFromCollection( RowCollection< ColumnCollection< double > > const&  rows )
{
    assert( rows.size() == N1 );

    assert
    (
        std::all_of
        ( 
            rows.begin(), 
            rows.end(), 
            []( auto const& column ){ return column.size() == N2; } 
        )
    );

    ZipIterator rowIter{ data.begin(), data.end(), rows.begin(), rows.end() };

    while ( !rowIter.isEnd() )
    {
        auto rowPair = (*rowIter);
        ZipIterator dataIter{ rowPair.first.begin(), rowPair.first.end(), rowPair.second.begin(), rowPair.second.end() };

        while( !dataIter.isEnd() )
        {
            (*dataIter).first = (*dataIter).second;

            ++dataIter;
        }

        ++rowIter;
    }
}

template< int N1, int N2 >
template< template< typename > class RowCollection, 
          template< typename > class ColumnCollection >
Matrix< N1, N2 >::Matrix( RowCollection< ColumnCollection< double > > const&  rows )
{
    createMatrixFromCollection( rows );
}

template< int N1, int N2 >
Matrix< N1, N2 >::Matrix( std::initializer_list< std::initializer_list< double > > const & rows )
{
    createMatrixFromCollection( rows );
}

template< int N1, int N2 >
Matrix< N1, N2 >::Matrix()
{
    for ( auto& row : data )
    {
        for ( auto& elem : row )
        {
            elem = 0.0;
        }
    }
}

template< int K1, int K2, typename Function >
constexpr Matrix< K1, K2 > matrixOperation( Matrix< K1, K2 > const& lhs, Matrix< K1, K2 > const& rhs, Function operation )
{
    Matrix< K1, K2 > result;

    for ( std::size_t i = 0; i < K1; ++i )
    {
        for ( std::size_t j = 0; j < K2; ++j )
        {
        result.data[ i ][ j ] = operation( lhs.data[ i ][ j ], rhs.data[ i ][ j ] );
        }
    }

    return result;
}

#endif