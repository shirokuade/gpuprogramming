#!/bin/bash
#==============================================================================================
# Simplified Metal build (without shader pre-compilation)
# Use this if you don't have Xcode Command Line Tools installed yet
#==============================================================================================

echo "Building Metal GPU Raytracer (simplified build)..."

# Note: This version compiles the shader at runtime instead of compile-time
# The shader will be loaded from raytracer.metal file directly

# Compile the application with Objective-C++
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
    echo "Note: This build loads the shader at runtime from raytracer.metal"
    echo "Make sure raytracer.metal is in the same directory when running."
    echo ""
else
    echo "Build failed!"
    exit 1
fi
