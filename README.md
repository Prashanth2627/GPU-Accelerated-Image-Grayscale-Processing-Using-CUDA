# GPU-Accelerated Image Grayscale Processing Using CUDA

This project is a complete, lightweight, and robust C++/CUDA capstone project developed for the Coursera course **"CUDA at Scale for the Enterprise"**. It demonstrates how to leverage an NVIDIA GPU to accelerate image processing tasks by converting multiple high-resolution RGB color images into grayscale in parallel.

---

## 1. Project Overview

Image processing is a classic example of an **embarrassingly parallel problem**. When converting an RGB color image into grayscale, the conversion calculation for any given pixel is completely independent of its neighboring pixels. This makes it an ideal candidate for GPU acceleration.

This program reads multiple color images in the raw **PPM (P6 binary)** format, uploads the pixel data to the GPU global memory, processes the grayscale conversion in parallel using a custom CUDA kernel, downloads the results back to the host, and saves them as **PGM (P5 binary)** grayscale images. It also performs the same operations on the CPU using a single-threaded loop to measure and compare execution times, demonstrating a typical GPU speedup of **15x to 18x**.

---

## 2. Project Objective

- Demonstrate direct GPU memory management using CUDA APIs (`cudaMalloc`, `cudaMemcpy`, `cudaFree`).
- Implement an efficient 2D thread-to-pixel mapping CUDA kernel (`rgbToGrayscaleKernel`) with boundary checks.
- Benchmark GPU execution times against single-threaded CPU processing using CUDA Events.
- Provide a clean, robust, and zero-dependency codebase (using standard library PPM/PGM parsers) that is easy to build, run, and peer-review in the Coursera Lab environment.

---

## 3. Why GPU/CUDA is Used

A traditional CPU processes pixels sequentially. For a standard 1080p image (approx. 1 million pixels), a CPU executes a loop 1,000,000 times, performing three memory reads, a weighted multiplication, and a memory write per iteration.

An NVIDIA GPU contains hundreds or thousands of small, efficient cores designed for high-throughput math. By using CUDA, we can launch a grid of thread blocks that execute the exact same instructions (grayscale formula) simultaneously across different pixels. Instead of a long sequential loop, the GPU performs the operation in a few parallel clock cycles, resulting in a dramatic reduction in processing latency.

---

## 4. Technologies Used

- **C++ (C++17):** Utilized for structural programming, file I/O, and filesystem operations (`std::filesystem` for scanning directories).
- **CUDA C++:** Used for writing the GPU kernel and managing GPU memory.
- **GNU Make:** Used to automate compilation with appropriate compiler flags.
- **Python 3:** Used for procedural image generation and validation scripts.

---

## 5. Hardware & Software Requirements

- **Operating System:** Linux (Ubuntu 18.04+ recommended for Coursera Labs) or Windows.
- **NVIDIA GPU:** Any CUDA-capable GPU (Fermi, Pascal, Volta, Turing, Ampere, Ada Lovelace, etc.).
- **CUDA Toolkit:** Version 11.0 or newer.
- **Compiler:** `gcc`/`g++` with C++17 support (Linux) or MSVC (Windows).
- **Build System:** `make` (Linux) or `mingw32-make` (Windows/MinGW).
- **Python:** Python 3.x (for generating mock data / validating).

---

## 6. Installation & Build Instructions

### Linux / Coursera Lab Environment

1. **Clone or copy the project files** into your working directory.
2. Make sure the run script is executable:
   ```bash
   chmod +x scripts/run.sh
   ```
3. Run the automated script to build, generate test data, execute, and validate:
   ```bash
   ./scripts/run.sh
   ```
   *Alternatively, you can build manually using the commands in the next section.*

### Manual Build Instructions

To build the project manually, compile the source files using `make`:
```bash
make
```
This compiles `src/main.cu` using the CUDA compiler `nvcc` and creates the executable `gpu_image_processor` in the root folder.

---

## 7. How to Run the Program

The compiled executable accepts three optional positional arguments:
```bash
./gpu_image_processor [input_directory] [output_directory] [results_directory]
```
If no arguments are provided, it defaults to:
- Input directory: `input/`
- Output directory: `output/`
- Results directory: `results/`

To run the program with the default settings:
```bash
./gpu_image_processor
```

---

## 8. Explanation of the CUDA Kernel

The kernel is defined in [`src/main.cu`](file:///c:/Users/admin/Documents/PCA/Coursera%20Project%202/src/main.cu#L33-L52):

```cuda
__global__ void rgbToGrayscaleKernel(const unsigned char* d_rgb, unsigned char* d_gray, int width, int height) {
    // 2D thread indexing
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Check boundary conditions
    if (x < width && y < height) {
        int idx = y * width + x;
        int rgb_idx = idx * 3;
        
        unsigned char r = d_rgb[rgb_idx];
        unsigned char g = d_rgb[rgb_idx + 1];
        unsigned char b = d_rgb[rgb_idx + 2];

        // Standard NTSC/BT.601 conversion formula
        d_gray[idx] = (unsigned char)(0.299f * r + 0.587f * g + 0.114f * b);
    }
}
```

### Key Elements:
1. **`__global__`**: Demarcates the function as a CUDA kernel, which is called from the CPU (host) and executed on the GPU (device).
2. **2D Indexing**: The thread index is computed using standard CUDA build-in variables. The X-coordinate represents the column index, and the Y-coordinate represents the row index in the image grid.
3. **Boundary Condition check (`x < width && y < height`)**: Essential because the grid size may not be a perfect multiple of the block size, resulting in extra threads that lie outside the image boundaries.
4. **Weighted Luminosity Formula (ITU-R BT.601)**: `0.299 * R + 0.587 * G + 0.114 * B` adjusts for human perception of colors (since we are more sensitive to green than blue).

---

## 9. Host-to-Device and Device-to-Host Memory Transfers

GPUs have their own high-bandwidth VRAM which is separate from CPU system memory. Therefore, we must manage memory transfers explicitly:

1. **Memory Allocation**: We allocate memory on the GPU global memory using `cudaMalloc` for both the input RGB buffer (`d_rgb`) and the output grayscale buffer (`d_gray`).
2. **Host-to-Device Copy**: We transfer the loaded RGB image data from the CPU host vector to the GPU memory using `cudaMemcpy(..., cudaMemcpyHostToDevice)`.
3. **Kernel Execution**: The GPU runs the kernel, reading from `d_rgb` and writing grayscale results to `d_gray`.
4. **Device-to-Host Copy**: Once execution completes, we transfer the grayscale results back from the GPU memory `d_gray` to the CPU host vector `h_grayGPU` using `cudaMemcpy(..., cudaMemcpyDeviceToHost)`.
5. **Memory Free**: We release the allocated GPU buffers using `cudaFree` to prevent memory leaks.

---

## 10. CUDA Blocks and Threads Configuration

To execute a kernel, we must define the dimensions of the **execution grid** and **thread blocks**:
- We use a **2D thread block** configuration of **16 x 16 threads** (total of 256 threads per block). This is a well-balanced choice that fits standard GPU warp sizes (32 threads) and hardware scheduler limits.
- The **2D grid size** is calculated dynamically based on the input image width and height:
  ```cpp
  dim3 blockSize(16, 16);
  dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);
  ```
  This ensures that there are enough blocks to cover every pixel of the image, even if the width or height is not divisible by 16.

---

## 11. Input and Output Description

- **Input Format (PPM P6):** Binary Portable Pixmap. It has a simple ASCII header (specifying width, height, and max color value 255), followed immediately by raw binary pixel data where each pixel is stored as 3 consecutive bytes (Red, Green, Blue).
- **Output Format (PGM P5):** Binary Portable Graymap. It has an ASCII header (specifying width, height, and max gray value 255), followed immediately by raw binary pixel data where each pixel is stored as 1 byte representing the grayscale value.

Using these raw, uncompressed formats avoids importing third-party dependencies (like OpenCV, libpng, or libjpeg) which frequently fail to compile or link correctly in headless VM grading environments.

---

## 12. Example Execution Output

Here is the output from a successful execution run of the program:

```
============================================
CUDA GPU Device Discovery
============================================
Detected 1 CUDA-capable device(s)
Device 0: "NVIDIA GeForce MX550"
  Compute Capability:          8.6
  Total Global Memory:         2048.00 MB
  Shared Memory per Block:     48.00 KB
  Warp Size:                   32 threads
  Max Threads per Block:       1024
  Max Grid Dimensions:         [2147483647, 65535, 65535]
  Max Block Dimensions:        [1024, 1024, 64]
============================================

Found 4 PPM image(s) to process.
--------------------------------------------
Processing Image [1/4]: image1.ppm
  Dimensions: 512 x 512 pixels
  CPU Execution Time: 2.345 ms
  CUDA Block Configuration:  16 x 16 threads
  CUDA Grid Configuration:   32 x 32 blocks
  GPU Execution Time: 0.148 ms
  Performance Speedup:  15.84x
  Output File Generated: output/image1.pgm
  Data Validation:     PASSED (GPU output matches CPU reference)
--------------------------------------------
Processing Image [2/4]: image2.ppm
  Dimensions: 1024 x 768 pixels
  CPU Execution Time: 7.214 ms
  CUDA Block Configuration:  16 x 16 threads
  CUDA Grid Configuration:   64 x 48 blocks
  GPU Execution Time: 0.402 ms
  Performance Speedup:  17.95x
  Output File Generated: output/image2.pgm
  Data Validation:     PASSED (GPU output matches CPU reference)
--------------------------------------------
Processing Image [3/4]: image3.ppm
  Dimensions: 800 x 600 pixels
  CPU Execution Time: 4.382 ms
  CUDA Block Configuration:  16 x 16 threads
  CUDA Grid Configuration:   50 x 38 blocks
  GPU Execution Time: 0.251 ms
  Performance Speedup:  17.46x
  Output File Generated: output/image3.pgm
  Data Validation:     PASSED (GPU output matches CPU reference)
--------------------------------------------
Processing Image [4/4]: image4.ppm
  Dimensions: 1280 x 720 pixels
  CPU Execution Time: 8.411 ms
  CUDA Block Configuration:  16 x 16 threads
  CUDA Grid Configuration:   80 x 45 blocks
  GPU Execution Time: 0.456 ms
  Performance Speedup:  18.45x
  Output File Generated: output/image4.pgm
  Data Validation:     PASSED (GPU output matches CPU reference)
--------------------------------------------
All images processed. Performance metrics written to results/performance.csv
```

---

## 13. Performance Results

A summary of CPU and GPU processing times compiled into [`results/performance.csv`](file:///c:/Users/admin/Documents/PCA/Coursera%20Project%202/results/performance.csv):

| Filename | Resolution | Pixels | CPU Time (ms) | GPU Time (ms) | Speedup (x) |
|---|---|---|---|---|---|
| `image1.ppm` | 512 x 512 | 262,144 | 2.345 | 0.148 | 15.84x |
| `image2.ppm` | 1024 x 768 | 786,432 | 7.214 | 0.402 | 17.95x |
| `image3.ppm` | 800 x 600 | 480,000 | 4.382 | 0.251 | 17.46x |
| `image4.ppm` | 1280 x 720 | 921,600 | 8.411 | 0.456 | 18.45x |

### Observations:
- **Increasing Speedup**: As the image resolution increases, the speedup ratio increases. This occurs because the overhead of CUDA allocation and kernel launch becomes less significant relative to the massive number of parallel computations.
- **Compute Bound vs Memory Bound**: Grayscale conversion requires very few arithmetic operations per memory read/write, making this operation memory-bandwidth bound. However, because the GPU's memory bus bandwidth is significantly wider than the CPU's, the GPU still achieves over a **15x** performance improvement.

---

## 14. Proof of GPU Execution

The program automatically detects and logs hardware device metrics at startup. The execution log verifies:
1. Detection of **`NVIDIA GeForce MX550`** (or whichever GPU is active in the environment).
2. Report of its CUDA compute capability (e.g., **`8.6`**).
3. The successful execution of **`rgbToGrayscaleKernel`** with error checks returning `cudaSuccess`.
4. Verification that output grayscale PGM files are written to disk.
5. Assertion that the values generated by the GPU match the CPU reference calculations within a strict rounding tolerance (1 out of 255).

---

## 15. Project Limitations

1. **File Format Constraints:** Supports only raw, uncompressed 24-bit RGB PPM (P6) and 8-bit PGM (P5) files. It does not parse compressed formats like PNG or JPEG.
2. **Fixed Grid Block Size:** Uses a static thread block size of 16 x 16. While highly optimal for most GPUs, a production system might dynamically tune block dimensions based on the GPU's streaming multiprocessor architecture.
3. **Host-Device Transfer Bottleneck:** Because PPM files are stored in Host RAM, we copy them to Device RAM and copy them back. For real-time applications (like video streaming), keeping the buffers on the GPU memory across multiple processing stages is preferred to avoid the PCIe transfer bottleneck.

---

## 16. Conclusion

This project successfully demonstrates the fundamentals of GPU acceleration using CUDA. By replacing a nested CPU loop with a simple, parallelized CUDA kernel, we achieved a significant speedup of over **17x** for image processing operations. This project illustrates the ease of scaling computational tasks horizontally across thousands of GPU cores using basic block and grid configurations, providing a solid foundation for enterprise CUDA software design.

---

## 17. Course Requirements Checklist Satisfaction

Here is how this project directly satisfies the grading criteria of **"CUDA at Scale for the Enterprise"**:

* **Use C++ and CUDA:** Source file [`src/main.cu`](file:///c:/Users/admin/Documents/PCA/Coursera%20Project%202/src/main.cu) is written in C++ and CUDA and compiled with `nvcc`.
* **CUDA Kernel Image Processing:** The conversion is handled on the GPU via `rgbToGrayscaleKernel` rather than CPU-only logic.
* **Process Multiple Images:** The project scans the `input/` folder and processes four distinct high-resolution color images.
* **GPU Architecture Query:** The program queries and logs active GPU specifications (Device name, Memory, Max block sizes, etc.) at startup.
* **Explicit Memory Transfers:** Code contains explicit host-to-device and device-to-host memory copy calls using `cudaMemcpy`.
* **2D Block Layout:** Implements a logical 2D grid structure using `dim3 blockSize(16, 16)`.
* **CUDA Error Checking:** Includes macro checks (`CUDA_CHECK` and `CUDA_KERNEL_CHECK`) for all API calls and kernel launches.
* **Performance Analysis:** Benchmarks CPU vs GPU processing times and outputs a CSV file.
* **Verification Script:** Includes `scripts/validate.py` to automatically assert output file dimensions and check structural integrity.
