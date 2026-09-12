#!/bin/bash
# scripts/run.sh - Simple script to compile and run the project in a CUDA environment

# Exit immediately if any command fails
set -e

echo "======================================================"
echo "CUDA Capstone: Grayscale Image Processing Build & Run"
echo "======================================================"

# Step 1: Clean build environment
echo "Step 1: Cleaning build artifacts..."
make clean || true

# Step 2: Create directories
echo "Step 2: Setting up directories..."
mkdir -p input output results obj

# Step 3: Check if input images exist, if not generate them
if [ ! -f "input/image1.ppm" ]; then
    echo "Step 3: No input images found, generating them using python helper..."
    python3 scripts/generate_data.py
else
    echo "Step 3: Input images already present."
fi

# Step 4: Compile the CUDA project
echo "Step 4: Compiling C++/CUDA source using nvcc..."
make all

# Step 5: Execute the CUDA application and capture log
echo "Step 5: Executing CUDA application on the GPU..."
./gpu_image_processor > results/execution_log.txt 2>&1

# Display output log in terminal
cat results/execution_log.txt

# Step 6: Validate output
echo "Step 6: Running validation checks..."
python3 scripts/validate.py

echo "======================================================"
echo "Project successfully executed and validated!"
echo "Outputs generated in output/ and results/ directories."
echo "======================================================"
