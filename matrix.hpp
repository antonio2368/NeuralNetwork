#ifndef MATRIX_H
#define MATRIX_

#include "zip.hpp"

#include <array>
#include <exception>
#include <algorithm>
#include <cassert>
#include <iostream>
#include <iomanip>

namespace
{
    constexpr int doublePrecision = 5;
    constexpr int outputWidth = 9;
}

template< int N1, int N2 >
class Matrix
{
    static_assert( N1 > 0 && N2 > 0, "Dimension must be positive integers." );

private:

    std::array< std::array< double, N2 >, N1 > data;

    template< template< typename > class RowCollection, 
              template< typename > class ColumnCollection >
    void createMatrixFromCollection( RowCollection< ColumnCollection< double > > const&  );
public:
    Matrix();

    template< template< typename > class RowCollection, 
              template< typename > class ColumnCollection >
    Matrix( RowCollection< ColumnCollection< double > > const& );

    Matrix( std::initializer_list< std::initializer_list< double > >const & );

    constexpr std::size_t getRowNum() const noexcept
    {
        return N1;
    }

    constexpr std::size_t getColNum() const noexcept
    {
        return N2;
    }

    double max() const noexcept;

    constexpr std::array< double, N2 > getRow( std::size_t index ) const noexcept;

    constexpr std::array< double, N1 > getColumn( std::size_t index ) const noexcept;

    template< std::size_t K >
    friend std::ostream& operator<<( std::ostream&, std::array< double, K > const& );

    template< int K1, int K2 >
    friend std::ostream& operator<<( std::ostream&, Matrix< K1, K2 > const& );
};

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

    ZipIterator rowIter{ data.begin(), rows.begin() };
    ZipIterator rowIterEnd{ data.end(), rows.end() };

    while ( rowIter != rowIterEnd )
    {
        auto rowPair = (*rowIter);
        ZipIterator dataIter{ rowPair.first.begin(), rowPair.second.begin() };
        ZipIterator dataIterEnd{ rowPair.first.end(), rowPair.second.end() };

        while( dataIter != dataIterEnd )
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

#endif