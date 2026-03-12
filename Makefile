BUILD_DIR := build
SRC_DIR := src
INC_DIR := include

CXX := amdclang++
OMP_OFFLOAD_ARCH ?= gfx942
CPPFLAGS := -I$(INC_DIR)
CXXFLAGS := -std=c++17 -O3 -g -fopenmp --offload-arch=$(OMP_OFFLOAD_ARCH) -fno-omit-frame-pointer -ffast-math
LDFLAGS := -fopenmp --offload-arch=$(OMP_OFFLOAD_ARCH)

TARGET := $(BUILD_DIR)/main
SRCS := $(SRC_DIR)/main.cpp $(SRC_DIR)/numerics.cpp
OBJS := $(BUILD_DIR)/main.o $(BUILD_DIR)/numerics.o

.PHONY: all run render clean

all: $(TARGET)

$(TARGET): $(OBJS) | $(BUILD_DIR)
	$(CXX) $(OBJS) $(LDFLAGS) -o $@

$(BUILD_DIR)/main.o: $(SRC_DIR)/main.cpp $(INC_DIR)/numerics.h | $(BUILD_DIR)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@

$(BUILD_DIR)/numerics.o: $(SRC_DIR)/numerics.cpp $(INC_DIR)/numerics.h | $(BUILD_DIR)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

sim: $(TARGET)
	./$(TARGET) $(ARGS)

clean:
	rm -f $(TARGET) $(OBJS) $(BUILD_DIR)/density.bin
