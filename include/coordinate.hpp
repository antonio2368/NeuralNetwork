#ifndef COORDINATE_H
#define COORDINATE_H

#include<iostream>

struct Coordinate
{
private:
    int first_;
    int second_;

public:
    Coordinate( int first, int second ) : first_( first ), second_( second ) {}

    int first() const noexcept
    {
        return first_;
    }

    int second() const noexcept
    {
        return second_;
    }

    friend std::ostream& operator<<( std::ostream& os, Coordinate const& coordinate );
};

#endif