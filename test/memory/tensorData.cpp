#include "memory/tensorData.hpp"
#include "gtest/gtest.h"

// TEST( tensorDataTest, indexOperator)
// {
//     nn::memory::TensorData< int, 5 > test;
//     ASSERT_EQ( typeid( int ), typeid( test[ 0 ] ) );

//     test[ 0 ] = 6;
    
//     ASSERT_EQ( test[ 0 ], 6 );
// }

// TEST( tensorDataTest, scalarValue )
// {
//     nn::memory::TensorData< int > data;
//     data = 10;

//     ASSERT_EQ( data.get(), 10 ); 
// }

// TEST( tensorDataTest, defaultInit )
// {
//     nn::memory::TensorData< double, 1 > test;
//     ASSERT_EQ( test[ 0 ], double{} );

//     nn::memory::TensorData< int, 1 > testInt;
//     ASSERT_EQ( testInt[ 0 ], int{} );

//     nn::memory::TensorData< int > scalarTest;
//     ASSERT_EQ( scalarTest.get(), int{} );
// }

// TEST( tensorDataTest, constructors )
// {
//     nn::memory::TensorData< int > scalar{ 12 };
//     ASSERT_EQ( scalar.get(), 12 );

//     nn::memory::TensorData< int, Dynamic > dynamic;
//     ASSERT_FALSE( dynamic.size() );
//     dynamic.setSize( 12 );
//     ASSERT_EQ( dynamic.size(), 12 );

// }