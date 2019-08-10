CXX=g++
BUILD=build
INCLUDE=includes
CFLAGS=-std=c++17

matrixTest: matrixTest.cpp $(BUILD)/matrix.o
			$(CXX) $(CFLAGS) -o $(BUILD)/$@ $^ -I$(INCLUDE)

%.o: %.cpp
	 $(CXX) -c $(CFLAGS) -o $(BUILD)/$@ $^
