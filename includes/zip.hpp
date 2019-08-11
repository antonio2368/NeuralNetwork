#ifndef ZIP_H
#define ZIP_H

#include <iterator>
#include <iostream>
#include <optional>

template< typename FirstIterator, typename SecondIterator > 
class ZipIterator
{
private:
    using T1 = typename std::iterator_traits< FirstIterator >::reference;
    using T2 = typename std::iterator_traits< SecondIterator >::reference;

    FirstIterator firstIt;
    FirstIterator firstItEnd;
    SecondIterator secondIt;
    SecondIterator secondItEnd;
    std::optional< std::pair< T1, T2 > > pair;

public:
    ZipIterator( FirstIterator const& firstIt, FirstIterator const& firstItEnd, 
                 SecondIterator const& secondIt, SecondIterator const& secondItEnd ) 
        : firstIt{ firstIt }, firstItEnd{ firstItEnd }, secondIt{ secondIt }, secondItEnd{ secondItEnd }
    {
    }

    bool operator==( ZipIterator< FirstIterator, SecondIterator > const& other )
    {
        return firstIt == other.firstIt && secondIt == other.secondIt;
    }

    bool operator!=( ZipIterator< FirstIterator, SecondIterator > const& other )
    {
        return !( *this == other );
    }

    ZipIterator& operator++()
    {
        ++firstIt;
        ++secondIt;

        pair.reset();

        return *this;
    }

    ZipIterator operator++( int )
    {
        ZipIterator temp{ *this };
        ++( *this );
        return temp;
    }

    auto operator*()
    {
        if ( !pair.has_value() )
        {
            pair.emplace( *firstIt, *secondIt );
        }

        return pair.value();
    }
    
    bool isEnd() const noexcept 
    {
        return firstIt == firstItEnd || secondIt == secondItEnd;
    }
};

#endif