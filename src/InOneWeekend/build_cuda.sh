#!/bin/bash
#==============================================================================================
# Build script for CUDA GPU Raytracer
#==============================================================================================

echo "Building CUDA GPU Raytracer..."

# Check if nvcc is available
if ! command -v nvcc &> /dev/null; then
    echo "Error: nvcc (CUDA compiler) not found"
    echo "Please install NVIDIA CUDA Toolkit"
    echo "Download from: https://developer.nvidia.com/cuda-downloads"
    exit 1
fi

# Check CUDA version
echo "CUDA Compiler version:"
nvcc --version | grep "release"
echo ""

# Compile CUDA program
echo "Compiling CUDA raytracer..."
nvcc -O3 \
    -arch=sm_50 \
    --use_fast_math \
    -Xcompiler -march=native \
    main_cuda.cu \
    -o rtow_cuda

if [ $? -eq 0 ]; then
    echo ""
    echo "Build successful!"
    echo ""
    echo "Run with: ./rtow_cuda > output_cuda.ppm"
    echo ""
    echo "Performance comparison:"
    echo "  CPU version: g++ -std=c++11 -O3 main.cc -o rtow_cpu && ./rtow_cpu > output_cpu.ppm"
    echo "  GPU version: ./rtow_cuda > output_gpu.ppm"
    echo ""
    echo "Expected speedup: 10-100x depending on GPU"
    echo ""
    echo "Note: Adjust -arch=sm_XX based on your GPU:"
    echo "  sm_50: Maxwell (GTX 900 series)"
    echo "  sm_60: Pascal (GTX 1000 series)"
    echo "  sm_70: Volta (Titan V)"
    echo "  sm_75: Turing (RTX 2000 series)"
    echo "  sm_80: Ampere (RTX 3000 series)"
    echo "  sm_86: Ampere (RTX 3050/3060)"
    echo "  sm_89: Ada Lovelace (RTX 4000 series)"
else
    echo ""
    echo "Build failed!"
    exit 1
fi
