#include "coordinate.hpp"

std::ostream& operator<<( std::ostream& os, Coordinate const& coordinate )
{
    os << '(' << coordinate.first() << ", " << coordinate.second() << ")";
    return os;
}