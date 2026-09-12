# Makefile for CUDA Image Grayscale Processing Capstone Project

# Compilers
NVCC = nvcc
CXX = g++

# Flags
# -O3 for optimizations, -std=c++17 for filesystem support, -arch=sm_60 for wide GPU compatibility
NVCC_FLAGS = -O3 -std=c++17 -arch=sm_60
CXX_FLAGS = -O3 -std=c++17 -Wall

# Paths
SRC_DIR = src
OBJ_DIR = obj
BIN = gpu_image_processor

# Object files
OBJS = $(OBJ_DIR)/main.o

# Rules
.PHONY: all clean run setup

all: setup $(BIN)

setup:
	@mkdir -p $(OBJ_DIR)
	@mkdir -p input output results

$(BIN): $(OBJS)
	$(NVCC) $(NVCC_FLAGS) -o $@ $^

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cu
	$(NVCC) $(NVCC_FLAGS) -c -o $@ $<

run: all
	./$(BIN)

clean:
	rm -rf $(OBJ_DIR) $(BIN) output/* results/performance.csv results/execution_log.txt
	@echo "Cleaned build artifacts and results."
