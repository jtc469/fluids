CXX := g++
CXXFLAGS = -std=c++17 -O3 -g -fno-omit-frame-pointer -fopenmp -I$(INC_DIR)
LDFLAGS := -fopenmp

BUILD_DIR := build
SRC_DIR := src
INC_DIR := include

TARGET := $(BUILD_DIR)/main
SRCS := $(SRC_DIR)/main.cpp $(SRC_DIR)/numerics.cpp
OBJS := $(BUILD_DIR)/main.o $(BUILD_DIR)/numerics.o

.PHONY: all run render clean

all: $(TARGET)

$(TARGET): $(OBJS) | $(BUILD_DIR)
	$(CXX) $(OBJS) $(LDFLAGS) -o $@

$(BUILD_DIR)/main.o: $(SRC_DIR)/main.cpp $(INC_DIR)/numerics.h | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILD_DIR)/numerics.o: $(SRC_DIR)/numerics.cpp $(INC_DIR)/numerics.h | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

sim: $(TARGET)
	./$(TARGET) $(ARGS)

run: $(TARGET)
	./$(TARGET) $(ARGS)
	python3 $(SRC_DIR)/render.py --out recent.gif --fps 60 --cmap Blues_r

render:
	python3 $(SRC_DIR)/render.py --out recent.gif --fps 60 --cmap Blues_r

clean:
	rm -f $(TARGET) $(OBJS) $(BUILD_DIR)/density.bin