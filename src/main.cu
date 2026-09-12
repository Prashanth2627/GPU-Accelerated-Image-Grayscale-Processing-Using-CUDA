#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <cctype>
#include <cuda_runtime.h>

namespace fs = std::filesystem;

// ============================================================================
// CUDA Error Checking Macros
// ============================================================================
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA Error in " << __FILE__ << " at line " << __LINE__ << ": " \
                      << cudaGetErrorString(err) << " (code: " << err << ")" << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

#define CUDA_KERNEL_CHECK() \
    do { \
        cudaError_t err = cudaGetLastError(); \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA Kernel Launch Error in " << __FILE__ << " at line " << __LINE__ << ": " \
                      << cudaGetErrorString(err) << " (code: " << err << ")" << std::endl; \
            exit(EXIT_FAILURE); \
        } \
        err = cudaDeviceSynchronize(); \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA Sync Error in " << __FILE__ << " at line " << __LINE__ << ": " \
                      << cudaGetErrorString(err) << " (code: " << err << ")" << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

// ============================================================================
// CUDA Kernel for Grayscale Conversion
// ============================================================================
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
        // Gray = 0.299 * R + 0.587 * G + 0.114 * B
        d_gray[idx] = (unsigned char)(0.299f * r + 0.587f * g + 0.114f * b);
    }
}

// ============================================================================
// CPU Reference implementation (Single Threaded)
// ============================================================================
void rgbToGrayscaleCPU(const unsigned char* rgb, unsigned char* gray, int width, int height) {
    for (int i = 0; i < width * height; ++i) {
        unsigned char r = rgb[i * 3];
        unsigned char g = rgb[i * 3 + 1];
        unsigned char b = rgb[i * 3 + 2];
        gray[i] = (unsigned char)(0.299f * r + 0.587f * g + 0.114f * b);
    }
}

// ============================================================================
// PPM (P6) / PGM (P5) Helper Functions
// ============================================================================
void skipComments(std::ifstream& ifs) {
    char c;
    while (ifs.get(c)) {
        if (std::isspace(c)) {
            continue;
        }
        if (c == '#') {
            std::string dummy;
            std::getline(ifs, dummy);
        } else {
            ifs.unget();
            break;
        }
    }
}

bool readPPM(const std::string& filename, int& width, int& height, int& max_val, std::vector<unsigned char>& data) {
    std::ifstream ifs(filename, std::ios::binary);
    if (!ifs.is_open()) {
        std::cerr << "Error: Could not open PPM file: " << filename << std::endl;
        return false;
    }

    std::string header;
    ifs >> header;
    if (header != "P6") {
        std::cerr << "Error: Invalid PPM header (must be P6): " << filename << std::endl;
        return false;
    }

    skipComments(ifs);
    ifs >> width;
    
    skipComments(ifs);
    ifs >> height;
    
    skipComments(ifs);
    ifs >> max_val;

    if (max_val != 255) {
        std::cerr << "Error: Only 8-bit PPM images (max value 255) are supported: " << filename << std::endl;
        return false;
    }

    // Consume the single whitespace character (usually a newline) before binary data
    ifs.get();

    int dataSize = width * height * 3;
    data.resize(dataSize);
    ifs.read(reinterpret_cast<char*>(data.data()), dataSize);

    if (ifs.gcount() != dataSize) {
        std::cerr << "Error: Failed to read pixel data from: " << filename << " (Expected " << dataSize << " bytes, read " << ifs.gcount() << ")" << std::endl;
        return false;
    }

    return true;
}

bool writePGM(const std::string& filename, int width, int height, int max_val, const std::vector<unsigned char>& data) {
    std::ofstream ofs(filename, std::ios::binary);
    if (!ofs.is_open()) {
        std::cerr << "Error: Could not open file for writing: " << filename << std::endl;
        return false;
    }

    ofs << "P5\n";
    ofs << "# Generated by CUDA Grayscale Converter\n";
    ofs << width << " " << height << "\n";
    ofs << max_val << "\n";

    int dataSize = width * height;
    ofs.write(reinterpret_cast<const char*>(data.data()), dataSize);

    return ofs.good();
}

// ============================================================================
// Device Property Query
// ============================================================================
void printDeviceProperties() {
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    
    std::cout << "============================================" << std::endl;
    std::cout << "CUDA GPU Device Discovery" << std::endl;
    std::cout << "============================================" << std::endl;
    
    if (err != cudaSuccess || deviceCount == 0) {
        std::cout << "No CUDA devices detected or drivers not configured properly." << std::endl;
        std::cout << "Error string: " << cudaGetErrorString(err) << std::endl;
        std::cout << "============================================" << std::endl << std::endl;
        return;
    }
    
    std::cout << "Detected " << deviceCount << " CUDA-capable device(s)" << std::endl;
    
    for (int dev = 0; dev < deviceCount; ++dev) {
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
        std::cout << "Device " << dev << ": \"" << prop.name << "\"" << std::endl;
        std::cout << "  Compute Capability:          " << prop.major << "." << prop.minor << std::endl;
        std::cout << "  Total Global Memory:         " << prop.totalGlobalMem / (1024.0 * 1024.0) << " MB" << std::endl;
        std::cout << "  Shared Memory per Block:     " << prop.sharedMemPerBlock / 1024.0 << " KB" << std::endl;
        std::cout << "  Warp Size:                   " << prop.warpSize << " threads" << std::endl;
        std::cout << "  Max Threads per Block:       " << prop.maxThreadsPerBlock << std::endl;
        std::cout << "  Max Grid Dimensions:         [" << prop.maxGridSize[0] << ", " 
                  << prop.maxGridSize[1] << ", " << prop.maxGridSize[2] << "]" << std::endl;
        std::cout << "  Max Block Dimensions:        [" << prop.maxThreadsDim[0] << ", " 
                  << prop.maxThreadsDim[1] << ", " << prop.maxThreadsDim[2] << "]" << std::endl;
    }
    std::cout << "============================================" << std::endl << std::endl;
}

// ============================================================================
// Main Execution
// ============================================================================
int main(int argc, char** argv) {
    std::string inputDir = "input";
    std::string outputDir = "output";
    std::string resultsDir = "results";

    // Set directories from command line if provided
    if (argc > 1) inputDir = argv[1];
    if (argc > 2) outputDir = argv[2];
    if (argc > 3) resultsDir = argv[3];

    // Ensure output directories exist
    if (!fs::exists(outputDir)) {
        fs::create_directories(outputDir);
    }
    if (!fs::exists(resultsDir)) {
        fs::create_directories(resultsDir);
    }

    // Print hardware specs
    printDeviceProperties();

    // Scan for PPM images in input directory
    std::vector<std::string> ppmFiles;
    if (!fs::exists(inputDir)) {
        std::cerr << "Error: Input directory '" << inputDir << "' does not exist." << std::endl;
        return 1;
    }

    for (const auto& entry : fs::directory_iterator(inputDir)) {
        if (entry.path().extension() == ".ppm") {
            ppmFiles.push_back(entry.path().string());
        }
    }

    if (ppmFiles.empty()) {
        std::cout << "No PPM files found in '" << inputDir << "'." << std::endl;
        std::cout << "Please populate '" << inputDir << "' with .ppm files to process." << std::endl;
        return 0;
    }

    std::cout << "Found " << ppmFiles.size() << " PPM image(s) to process." << std::endl;
    std::cout << "--------------------------------------------" << std::endl;

    // Open performance CSV
    std::string csvPath = resultsDir + "/performance.csv";
    std::ofstream csvFile(csvPath);
    if (!csvFile.is_open()) {
        std::cerr << "Error: Could not create performance CSV at: " << csvPath << std::endl;
        return 1;
    }
    csvFile << "filename,width,height,cpu_time_ms,gpu_time_ms,speedup\n";

    int processedCount = 0;

    for (const auto& filePath : ppmFiles) {
        fs::path p(filePath);
        std::string filename = p.filename().string();
        std::cout << "Processing Image [" << ++processedCount << "/" << ppmFiles.size() << "]: " << filename << std::endl;

        int width = 0, height = 0, max_val = 0;
        std::vector<unsigned char> h_rgb;
        
        if (!readPPM(filePath, width, height, max_val, h_rgb)) {
            std::cerr << "  Skipping file due to read error." << std::endl;
            continue;
        }

        std::cout << "  Dimensions: " << width << " x " << height << " pixels" << std::endl;

        int pixelCount = width * height;
        size_t rgbSize = pixelCount * 3 * sizeof(unsigned char);
        size_t graySize = pixelCount * 1 * sizeof(unsigned char);

        std::vector<unsigned char> h_grayCPU(pixelCount);
        std::vector<unsigned char> h_grayGPU(pixelCount);

        // 1. CPU REFERENCE EXECUTION
        auto cpuStart = std::chrono::high_resolution_clock::now();
        rgbToGrayscaleCPU(h_rgb.data(), h_grayCPU.data(), width, height);
        auto cpuEnd = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> cpuDuration = cpuEnd - cpuStart;
        double cpuTimeMs = cpuDuration.count();
        std::cout << "  CPU Execution Time: " << cpuTimeMs << " ms" << std::endl;

        // 2. GPU CUDA EXECUTION
        unsigned char* d_rgb = nullptr;
        unsigned char* d_gray = nullptr;

        // Allocate memory on device
        CUDA_CHECK(cudaMalloc(&d_rgb, rgbSize));
        CUDA_CHECK(cudaMalloc(&d_gray, graySize));

        // Copy input image data from host to device
        CUDA_CHECK(cudaMemcpy(d_rgb, h_rgb.data(), rgbSize, cudaMemcpyHostToDevice));

        // Define grid and block structure (2D)
        dim3 blockSize(16, 16);
        dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);
        
        std::cout << "  CUDA Block Configuration:  " << blockSize.x << " x " << blockSize.y << " threads" << std::endl;
        std::cout << "  CUDA Grid Configuration:   " << gridSize.x << " x " << gridSize.y << " blocks" << std::endl;

        // CUDA Timing Events
        cudaEvent_t startEvent, stopEvent;
        CUDA_CHECK(cudaEventCreate(&startEvent));
        CUDA_CHECK(cudaEventCreate(&stopEvent));

        // Record start event
        CUDA_CHECK(cudaEventRecord(startEvent, 0));

        // Launch Kernel
        rgbToGrayscaleKernel<<<gridSize, blockSize>>>(d_rgb, d_gray, width, height);
        
        // Kernel Check & Synchronize
        CUDA_KERNEL_CHECK();

        // Record stop event
        CUDA_CHECK(cudaEventRecord(stopEvent, 0));
        CUDA_CHECK(cudaEventSynchronize(stopEvent));

        // Calculate elapsed time
        float gpuTimeMs = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&gpuTimeMs, startEvent, stopEvent));
        std::cout << "  GPU Execution Time: " << gpuTimeMs << " ms" << std::endl;

        // Compute performance comparison
        double speedup = cpuTimeMs / gpuTimeMs;
        std::cout << "  Performance Speedup:  " << speedup << "x" << std::endl;

        // Copy grayscale result from device to host
        CUDA_CHECK(cudaMemcpy(h_grayGPU.data(), d_gray, graySize, cudaMemcpyDeviceToHost));

        // Clean up GPU resources
        CUDA_CHECK(cudaEventDestroy(startEvent));
        CUDA_CHECK(cudaEventDestroy(stopEvent));
        CUDA_CHECK(cudaFree(d_rgb));
        CUDA_CHECK(cudaFree(d_gray));

        // Save output grayscale PGM image
        std::string outFilename = outputDir + "/" + p.stem().string() + ".pgm";
        if (writePGM(outFilename, width, height, max_val, h_grayGPU)) {
            std::cout << "  Output File Generated: " << outFilename << std::endl;
        } else {
            std::cerr << "  Failed to write output image." << std::endl;
        }

        // Validate correctness of GPU output against CPU reference
        bool correct = true;
        for (int i = 0; i < pixelCount; ++i) {
            if (std::abs(h_grayGPU[i] - h_grayCPU[i]) > 1) { // Tolerance of 1 to allow slight floating point rounding discrepancies
                correct = false;
                break;
            }
        }
        std::cout << "  Data Validation:     " << (correct ? "PASSED (GPU output matches CPU reference)" : "FAILED (Mismatch detected)") << std::endl;

        // Write metrics to CSV
        csvFile << filename << "," << width << "," << height << "," << cpuTimeMs << "," << gpuTimeMs << "," << speedup << "\n";
        
        std::cout << "--------------------------------------------" << std::endl;
    }

    csvFile.close();
    std::cout << "All images processed. Performance metrics written to " << csvPath << std::endl;
    return 0;
}
