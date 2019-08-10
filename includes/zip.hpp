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
    SecondIterator secondIt;
    std::optional< std::pair< T1, T2 > > pair;

public:
    ZipIterator( FirstIterator const& firstIt, SecondIterator const& secondIt ) 
        : firstIt{ firstIt }, secondIt{ secondIt }
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
};

#endif