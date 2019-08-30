CXX=g++
BUILD=build
INCLUDE=includes
CFLAGS=-std=c++17

$(BUILD)/matrixTest: matrixTest.cpp $(BUILD)/matrix.o matrix.hpp $(BUILD)/coordinate.o
			$(CXX) $(CFLAGS) -o $@ $^ -I$(INCLUDE)

$(BUILD)/%.o: %.cpp
	 $(CXX) -c $(CFLAGS) -o $@ $^ -I$(INCLUDE)
