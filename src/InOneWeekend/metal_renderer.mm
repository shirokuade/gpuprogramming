//==============================================================================================
// Metal Renderer Bridge - Objective-C++ wrapper for Metal API
//==============================================================================================

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <iostream>
#include <vector>
#include <memory>
#include "rtweekend.h"
#include "hittable_list.h"
#include "sphere.h"
#include "material.h"
#include "camera.h"

// GPU-compatible structures matching Metal shader
struct GPUSphere {
    float center[3];
    float radius;
    int material_type;
    float albedo[3];
    float fuzz;
    float refraction_index;
};

struct GPUCamera {
    float pixel00_loc[3];
    float pixel_delta_u[3];
    float pixel_delta_v[3];
    float center[3];
    float defocus_disk_u[3];
    float defocus_disk_v[3];
    float defocus_angle;
};

class MetalRenderer {
private:
    id<MTLDevice> device;
    id<MTLCommandQueue> commandQueue;
    id<MTLComputePipelineState> pipelineState;
    id<MTLLibrary> library;

public:
    MetalRenderer() {
        // Get default Metal device (M3 GPU)
        device = MTLCreateSystemDefaultDevice();
        if (!device) {
            std::cerr << "Metal is not supported on this device\n";
            exit(1);
        }

        std::cout << "Using Metal device: " << [[device name] UTF8String] << "\n";

        // Create command queue
        commandQueue = [device newCommandQueue];

        // Load Metal shader library
        NSError* error = nil;
        NSString* shaderPath = @"raytracer.metal";
        NSString* shaderSource = [NSString stringWithContentsOfFile:shaderPath
                                                            encoding:NSUTF8StringEncoding
                                                               error:&error];

        if (!shaderSource) {
            // Try loading from current directory
            const char* currentPath = "raytracer.metal";
            std::ifstream file(currentPath);
            if (!file.good()) {
                std::cerr << "Error: Could not find raytracer.metal shader file\n";
                exit(1);
            }
            file.close();

            shaderSource = [NSString stringWithContentsOfFile:@(currentPath)
                                                      encoding:NSUTF8StringEncoding
                                                         error:&error];
        }

        library = [device newLibraryWithSource:shaderSource options:nil error:&error];
        if (!library) {
            std::cerr << "Error loading shader library: "
                      << [[error localizedDescription] UTF8String] << "\n";
            exit(1);
        }

        // Get kernel function
        id<MTLFunction> kernelFunction = [library newFunctionWithName:@"raytrace_kernel"];
        if (!kernelFunction) {
            std::cerr << "Error: Could not find raytrace_kernel function\n";
            exit(1);
        }

        // Create compute pipeline
        pipelineState = [device newComputePipelineStateWithFunction:kernelFunction error:&error];
        if (!pipelineState) {
            std::cerr << "Error creating pipeline: "
                      << [[error localizedDescription] UTF8String] << "\n";
            exit(1);
        }

        std::cout << "Metal renderer initialized successfully\n";
    }

    void render(const hittable_list& world, const camera& cam,
                int image_width, int image_height, int samples, int depth) {

        std::cout << "Starting Metal GPU render...\n";
        std::cout << "Resolution: " << image_width << "x" << image_height << "\n";
        std::cout << "Samples: " << samples << ", Depth: " << depth << "\n";

        // Convert spheres to GPU format
        std::vector<GPUSphere> gpu_spheres;
        // Note: This is a simplified version. In practice, you'd need to extract
        // sphere data from the hittable_list

        // For now, we'll create the scene directly (matching main.cc)
        // In a production version, you'd parse the hittable_list

        // Create buffers
        int sphere_count = 0; // Will be filled when we parse the world
        id<MTLBuffer> sphereBuffer = [device newBufferWithLength:sizeof(GPUSphere) * 500
                                                         options:MTLResourceStorageModeShared];

        // Camera buffer
        GPUCamera gpu_camera;
        // Note: You'd need to extract these from the camera object
        // This is simplified for the example
        id<MTLBuffer> cameraBuffer = [device newBufferWithBytes:&gpu_camera
                                                          length:sizeof(GPUCamera)
                                                         options:MTLResourceStorageModeShared];

        // Parameters
        id<MTLBuffer> sphereCountBuffer = [device newBufferWithBytes:&sphere_count
                                                              length:sizeof(int)
                                                             options:MTLResourceStorageModeShared];

        id<MTLBuffer> samplesBuffer = [device newBufferWithBytes:&samples
                                                          length:sizeof(int)
                                                         options:MTLResourceStorageModeShared];

        id<MTLBuffer> depthBuffer = [device newBufferWithBytes:&depth
                                                        length:sizeof(int)
                                                       options:MTLResourceStorageModeShared];

        // Output buffer
        int pixel_count = image_width * image_height;
        id<MTLBuffer> outputBuffer = [device newBufferWithLength:sizeof(float) * 3 * pixel_count
                                                         options:MTLResourceStorageModeShared];

        // Create command buffer
        id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];

        [encoder setComputePipelineState:pipelineState];
        [encoder setBuffer:sphereBuffer offset:0 atIndex:0];
        [encoder setBuffer:cameraBuffer offset:0 atIndex:1];
        [encoder setBuffer:sphereCountBuffer offset:0 atIndex:2];
        [encoder setBuffer:samplesBuffer offset:0 atIndex:3];
        [encoder setBuffer:depthBuffer offset:0 atIndex:4];
        [encoder setBuffer:outputBuffer offset:0 atIndex:5];

        // Dispatch threads
        MTLSize gridSize = MTLSizeMake(image_width, image_height, 1);
        NSUInteger threadGroupSize = pipelineState.maxTotalThreadsPerThreadgroup;
        NSUInteger threadGroupWidth = 16;
        NSUInteger threadGroupHeight = threadGroupSize / threadGroupWidth;
        MTLSize threadgroupSize = MTLSizeMake(threadGroupWidth, threadGroupHeight, 1);

        [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
        [encoder endEncoding];

        // Execute
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];

        std::cout << "GPU render completed!\n";

        // Read back results and output PPM
        float* output = (float*)[outputBuffer contents];

        std::cout << "P3\n" << image_width << ' ' << image_height << "\n255\n";

        for (int j = 0; j < image_height; j++) {
            for (int i = 0; i < image_width; i++) {
                int idx = (j * image_width + i) * 3;

                // Gamma correction (sqrt for gamma 2)
                float r = sqrt(output[idx + 0]);
                float g = sqrt(output[idx + 1]);
                float b = sqrt(output[idx + 2]);

                // Clamp and convert to 0-255
                int ir = int(256 * std::clamp(r, 0.0f, 0.999f));
                int ig = int(256 * std::clamp(g, 0.0f, 0.999f));
                int ib = int(256 * std::clamp(b, 0.0f, 0.999f));

                std::cout << ir << ' ' << ig << ' ' << ib << '\n';
            }
        }

        std::cerr << "\rDone.                 \n";
    }

    ~MetalRenderer() {
        [device release];
        [commandQueue release];
        [pipelineState release];
        [library release];
    }
};

// C interface for use in main
extern "C" void render_with_metal(const hittable_list& world, const camera& cam,
                                   int width, int height, int samples, int depth) {
    MetalRenderer renderer;
    renderer.render(world, cam, width, height, samples, depth);
}
