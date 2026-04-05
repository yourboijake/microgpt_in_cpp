# Variables to make life easier
CXX = g++
CXXFLAGS = -std=c++17 -Wall
SRCS = $(wildcard *.cpp)
TARGET = my_program

# The default rule
all:
	$(CXX) $(CXXFLAGS) $(SRCS) -o $(TARGET)

# Clean rule to remove the executable
clean:
	rm -f $(TARGET)

