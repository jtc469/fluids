CXX := g++
MPICXX ?= mpic++
GPU_CXX ?= $(CXX)

# Default to an aggressive high-performance native build. Users can override CXXFLAGS/LDFLAGS.
CXXFLAGS ?= -std=c++17 -Ofast -DNDEBUG -march=native -ffast-math -fno-math-errno -fno-trapping-math -funroll-loops -fomit-frame-pointer -fopenmp -fopenmp-simd -flto -Iinclude
LDFLAGS ?= -fopenmp -flto

# OpenMP runtime defaults that are usually better on multi-socket/NUMA nodes.
OMP_PROC_BIND ?= close
OMP_PLACES ?= cores
OMP_NUM_THREADS ?= 4
OMP_ENV := OMP_PROC_BIND=$(OMP_PROC_BIND) OMP_PLACES=$(OMP_PLACES)
ifneq ($(strip $(OMP_NUM_THREADS)),)
OMP_ENV += OMP_NUM_THREADS=$(OMP_NUM_THREADS)
endif

BUILD_DIR := build
SRC_DIR := src
INC_DIR := include

TARGET := $(BUILD_DIR)/main
MPI_TARGET := $(BUILD_DIR)/main_mpi
GPU_TARGET := $(BUILD_DIR)/main_gpu
SRCS := $(SRC_DIR)/main.cpp $(SRC_DIR)/numerics.cpp
OBJS := $(BUILD_DIR)/main.o $(BUILD_DIR)/numerics.o
MPI_OBJS := $(BUILD_DIR)/main_mpi.o $(BUILD_DIR)/numerics_mpi.o
GPU_OBJS := $(BUILD_DIR)/main_gpu.o $(BUILD_DIR)/numerics_gpu.o

MPI_CXXFLAGS := $(CXXFLAGS) -DUSE_MPI
GPU_OFFLOAD_FLAGS ?=
GPU_CXXFLAGS ?= -std=c++17 -Ofast -DNDEBUG -march=native -ffast-math -fopenmp -fopenmp-simd -DUSE_OMP_TARGET $(GPU_OFFLOAD_FLAGS) -Iinclude
GPU_LDFLAGS ?= -fopenmp $(GPU_OFFLOAD_FLAGS)
NP ?= 2

.PHONY: all run render clean sim debug release native mpi sim-mpi gpu sim-gpu

all: $(TARGET)

$(TARGET): $(OBJS) | $(BUILD_DIR)
	$(CXX) $(OBJS) $(LDFLAGS) -o $@

$(MPI_TARGET): $(MPI_OBJS) | $(BUILD_DIR)
	$(MPICXX) $(MPI_OBJS) $(LDFLAGS) -o $@

$(GPU_TARGET): $(GPU_OBJS) | $(BUILD_DIR)
	$(GPU_CXX) $(GPU_OBJS) $(GPU_LDFLAGS) -o $@

$(BUILD_DIR)/main.o: $(SRC_DIR)/main.cpp $(INC_DIR)/numerics.h | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILD_DIR)/numerics.o: $(SRC_DIR)/numerics.cpp $(INC_DIR)/numerics.h | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILD_DIR)/main_mpi.o: $(SRC_DIR)/main.cpp $(INC_DIR)/numerics.h | $(BUILD_DIR)
	$(MPICXX) $(MPI_CXXFLAGS) -c $< -o $@

$(BUILD_DIR)/numerics_mpi.o: $(SRC_DIR)/numerics.cpp $(INC_DIR)/numerics.h | $(BUILD_DIR)
	$(MPICXX) $(MPI_CXXFLAGS) -c $< -o $@

$(BUILD_DIR)/main_gpu.o: $(SRC_DIR)/main.cpp $(INC_DIR)/numerics.h | $(BUILD_DIR)
	$(GPU_CXX) $(GPU_CXXFLAGS) -c $< -o $@

$(BUILD_DIR)/numerics_gpu.o: $(SRC_DIR)/numerics.cpp $(INC_DIR)/numerics.h | $(BUILD_DIR)
	$(GPU_CXX) $(GPU_CXXFLAGS) -c $< -o $@

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

sim: $(TARGET)
	$(OMP_ENV) ./$(TARGET) $(ARGS)

run: $(TARGET)
	$(OMP_ENV) ./$(TARGET) $(ARGS)
	python3 $(SRC_DIR)/render.py --out recent.gif --fps 60 --cmap Blues_r

mpi: $(MPI_TARGET)

sim-mpi: $(MPI_TARGET)
	mpirun -np $(NP) ./$(MPI_TARGET) $(ARGS)

gpu: $(GPU_TARGET)

sim-gpu: $(GPU_TARGET)
	$(OMP_ENV) ./$(GPU_TARGET) $(ARGS)

debug: CXXFLAGS := -std=c++17 -O0 -g -fopenmp -Iinclude
debug: clean all

release: CXXFLAGS := -std=c++17 -O3 -DNDEBUG -fopenmp -Iinclude
release: clean all

native: CXXFLAGS := -std=c++17 -Ofast -DNDEBUG -march=native -ffast-math -fno-math-errno -fno-trapping-math -funroll-loops -fomit-frame-pointer -fopenmp -fopenmp-simd -flto -Iinclude
native: LDFLAGS := -fopenmp -flto
native: clean all

render:
	python3 $(SRC_DIR)/render.py --out recent.gif --fps 60 --cmap Blues_r

clean:
	rm -f $(TARGET) $(MPI_TARGET) $(GPU_TARGET) $(OBJS) $(MPI_OBJS) $(GPU_OBJS) $(BUILD_DIR)/density.bin $(BUILD_DIR)/density_rank*.bin