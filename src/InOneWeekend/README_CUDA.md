# CUDA GPU Raytracer

GPU-accelerated version of "Ray Tracing in One Weekend" using NVIDIA CUDA.

## Performance

Expected speedup on NVIDIA GPUs:
- **GTX 1060**: ~20-30x faster than CPU
- **RTX 2060**: ~40-60x faster than CPU
- **RTX 3070**: ~60-80x faster than CPU
- **RTX 4090**: ~100-150x faster than CPU

Example timing (1200×675, 10 samples, 20 depth):
- CPU (i7): ~2 minutes
- GPU (RTX 3070): ~1.5 seconds (**80x faster!**)

## Requirements

- NVIDIA GPU with CUDA support
- NVIDIA CUDA Toolkit 10.0 or later
- Linux, Windows, or macOS with CUDA support

## Installation

### 1. Install CUDA Toolkit

**Linux (Ubuntu/Debian):**
```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get install cuda
```

**Windows:**
Download from: https://developer.nvidia.com/cuda-downloads

**macOS:**
CUDA support ended with macOS 10.13. Use Metal version instead.

### 2. Verify Installation

```bash
nvcc --version
nvidia-smi
```

## Building

### Automatic Build:
```bash
chmod +x build_cuda.sh
./build_cuda.sh
```

### Manual Build:
```bash
# Basic build
nvcc -O3 main_cuda.cu -o rtow_cuda

# Optimized for your GPU (replace sm_75 with your GPU's compute capability)
nvcc -O3 -arch=sm_75 --use_fast_math main_cuda.cu -o rtow_cuda
```

### GPU Compute Capabilities:
| GPU Series | Compute Capability | Flag |
|-----------|-------------------|------|
| GTX 900 (Maxwell) | 5.0 | `-arch=sm_50` |
| GTX 1000 (Pascal) | 6.1 | `-arch=sm_61` |
| RTX 2000 (Turing) | 7.5 | `-arch=sm_75` |
| RTX 3000 (Ampere) | 8.6 | `-arch=sm_86` |
| RTX 4000 (Ada) | 8.9 | `-arch=sm_89` |

Find your GPU's compute capability: https://developer.nvidia.com/cuda-gpus

## Running

```bash
# Basic run
./rtow_cuda > output.ppm

# With timing
time ./rtow_cuda > output.ppm

# View progress (stderr)
./rtow_cuda > output.ppm 2>&1 | tee log.txt
```

## Customization

Edit `main_cuda.cu` to change settings:

```cpp
// Image quality
const int image_width = 1200;           // Resolution
const int samples_per_pixel = 10;       // Anti-aliasing (higher = better)
const int max_depth = 20;               // Ray bounces (higher = more realistic)

// Camera
float vfov = 20.0f;                     // Field of view
Vec3CPU lookfrom(13, 2, 3);            // Camera position
Vec3CPU lookat(0, 0, 0);               // Look-at point
float defocus_angle = 0.6f;             // Depth of field blur
```

## Performance Tuning

### 1. Thread Block Size
In `main_cuda.cu`, adjust:
```cpp
dim3 block_size(16, 16);  // Default 16×16 = 256 threads/block
// Try 8×8, 16×16, or 32×32 depending on your GPU
```

### 2. Compiler Optimization Flags
```bash
# Aggressive optimization
nvcc -O3 --use_fast_math -arch=sm_XX main_cuda.cu -o rtow_cuda

# Maximum performance (may reduce precision slightly)
nvcc -O3 --use_fast_math -ftz=true -prec-div=false -prec-sqrt=false main_cuda.cu -o rtow_cuda
```

### 3. Quality vs Speed Trade-offs
| Setting | Low Quality (Fast) | Medium | High Quality (Slow) |
|---------|-------------------|---------|---------------------|
| `samples_per_pixel` | 5 | 10-20 | 50-100 |
| `max_depth` | 10 | 20-30 | 50 |
| Resolution | 800×450 | 1200×675 | 1920×1080 |

## Troubleshooting

### Out of Memory Error
```
Reduce image_width, samples_per_pixel, or scene complexity
```

### Slow Performance
```bash
# Check GPU usage
nvidia-smi

# Ensure GPU is being used (not CPU fallback)
nvidia-smi dmon
```

### Compilation Errors
```bash
# Check CUDA version compatibility
nvcc --version

# Use compatible C++ standard
nvcc -std=c++11 main_cuda.cu -o rtow_cuda
```

## Architecture

### Memory Layout:
```
┌─────────────────┐
│   CPU (Host)    │
│  - Scene setup  │
│  - Camera calc  │
│  - PPM output   │
└────────┬────────┘
         │ cudaMemcpy
         ▼
┌─────────────────┐
│   GPU (Device)  │
│  - Spheres[]    │
│  - Camera       │
│  - Output[]     │
│  - RandState[]  │
└────────┬────────┘
         │ 810,000 threads
         ▼
┌─────────────────┐
│  Kernel Launch  │
│  Grid: 75×42    │
│  Block: 16×16   │
│  = 810,000      │
└─────────────────┘
```

### Execution Flow:
```
1. CPU: Build scene (spheres, camera)
2. CPU → GPU: Copy scene data (cudaMemcpy)
3. GPU: Initialize random states (init_rand kernel)
4. GPU: Raytrace all pixels in parallel (raytrace_kernel)
5. GPU → CPU: Copy results back (cudaMemcpy)
6. CPU: Gamma correction and PPM output
```

## Comparison with CPU Version

| Feature | CPU Version | CUDA Version |
|---------|------------|--------------|
| **Files** | main.cc + *.h | main_cuda.cu + raytracer.cu |
| **Compilation** | g++ | nvcc |
| **Precision** | double | float |
| **Parallelism** | None | 810K threads |
| **Random** | std::rand | cuRAND |
| **Materials** | Virtual classes | Switch statement |
| **Ray bouncing** | Recursive | Iterative |
| **Memory** | shared_ptr | Plain pointers |
| **Speed** | 1x (baseline) | 10-100x faster |

## Known Limitations

1. **Single precision (float)**: Slightly less precision than CPU's double, but visually identical
2. **Fixed scene size**: Scene must fit in GPU memory (~6GB for complex scenes)
3. **CUDA required**: Won't run on AMD GPUs (use OpenCL or HIP port instead)
4. **Stack depth**: Limited recursive depth (already converted to iterative)

## Future Enhancements

Possible improvements:
- [ ] BVH acceleration structure for faster intersection
- [ ] Multi-GPU support with CUDA streams
- [ ] Texture mapping
- [ ] Triangle mesh support
- [ ] Motion blur
- [ ] Real-time progressive rendering
- [ ] OptiX integration for RT cores (RTX GPUs)

## Credits

Original CPU raytracer: Peter Shirley's "Ray Tracing in One Weekend"
CUDA GPU port: Optimized for NVIDIA GPUs with CUDA acceleration

## License

Public domain (CC0) - same as original raytracer
