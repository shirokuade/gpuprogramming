#!/bin/bash
#==============================================================================================
# Build script for Metal GPU Raytracer (Apple Silicon M3)
#==============================================================================================

echo "Building Metal GPU Raytracer for Apple Silicon..."

# Check if running on macOS
if [[ "$OSTYPE" != "darwin"* ]]; then
    echo "Error: Metal is only available on macOS"
    exit 1
fi

# Compile Metal shader to .metallib
echo "Compiling Metal shaders..."
xcrun -sdk macosx metal -c raytracer.metal -o raytracer.air
xcrun -sdk macosx metallib raytracer.air -o raytracer.metallib

# Compile the full application with Objective-C++
echo "Compiling application..."
clang++ -std=c++17 \
    -O3 \
    -march=native \
    main_metal.cc \
    metal_renderer.mm \
    -framework Metal \
    -framework Foundation \
    -framework CoreGraphics \
    -o rtow_metal

if [ $? -eq 0 ]; then
    echo "Build successful!"
    echo ""
    echo "Run with: ./rtow_metal > output.ppm"
    echo ""
    echo "Performance comparison:"
    echo "  CPU version: ./rtow > output_cpu.ppm"
    echo "  GPU version: ./rtow_metal > output_gpu.ppm"
    echo ""
    echo "Expected speedup on M3: 10-50x faster depending on scene complexity"
else
    echo "Build failed!"
    exit 1
fi
