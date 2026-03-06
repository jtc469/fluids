CXX := g++
CXXFLAGS := -std=c++17 -O2 -fopenmp
LDFLAGS := -fopenmp

TARGET := main
SRCS := main.cpp numerics.cpp
OBJS := $(SRCS:.cpp=.o)

.PHONY: all run render open clean

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(OBJS) $(LDFLAGS) -o $@

%.o: %.cpp numerics.h
	$(CXX) $(CXXFLAGS) -c $< -o $@

run: $(TARGET)
	./$(TARGET) $(ARGS)
	python3 render.py --out recent.gif --fps 120 --cmap Blues_r

clean:
	rm -f $(TARGET) $(OBJS)