#!/bin/bash
#==============================================================================================
# Build script for FULL Metal GPU Raytracer (Apple Silicon M3)
#==============================================================================================

echo "Building Full Metal GPU Raytracer for Apple Silicon..."

# Check if running on macOS
if [[ "$OSTYPE" != "darwin"* ]]; then
    echo "Error: Metal is only available on macOS"
    exit 1
fi

# Compile the full GPU implementation
echo "Compiling GPU application..."
clang++ -std=c++17 \
    -O3 \
    -march=native \
    main_metal_gpu.mm \
    -framework Metal \
    -framework Foundation \
    -framework CoreGraphics \
    -o rtow_gpu

if [ $? -eq 0 ]; then
    echo "Build successful!"
    echo ""
    echo "Run GPU version: ./rtow_gpu > output_gpu.ppm"
    echo ""
    echo "The shader will be compiled at runtime from raytracer.metal"
    echo "Make sure raytracer.metal is in the same directory."
    echo ""
    echo "Expected performance on M3: 10-50x faster than CPU"
else
    echo "Build failed!"
    exit 1
fi
