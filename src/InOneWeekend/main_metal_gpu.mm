//==============================================================================================
// Ray Tracing in One Weekend - Full Metal GPU Implementation
// Optimized for Apple Silicon M3
//==============================================================================================

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <iostream>
#include <vector>
#include <cmath>

// Undefine deprecated macOS pi constant
#ifdef pi
#undef pi
#endif

#include "rtweekend.h"
#include "hittable.h"
#include "hittable_list.h"
#include "material.h"
#include "sphere.h"

// Forward declare camera - we'll manually calculate its parameters
#include "vec3.h"
#include "ray.h"

// GPU-compatible structures
struct GPUSphere {
    float center[3];
    float radius;
    int material_type;  // 0=lambertian, 1=metal, 2=dielectric
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

void convert_vec3(const vec3& v, float out[3]) {
    out[0] = v.x();
    out[1] = v.y();
    out[2] = v.z();
}

int main() {
    // Build scene
    hittable_list world;

    auto ground_material = make_shared<lambertian>(color(0.5, 0.5, 0.5));
    world.add(make_shared<sphere>(point3(0,-1000,0), 1000, ground_material));

    for (int a = -11; a < 11; a++) {
        for (int b = -11; b < 11; b++) {
            auto choose_mat = random_double();
            point3 center(a + 0.9*random_double(), 0.2, b + 0.9*random_double());

            if ((center - point3(4, 0.2, 0)).length() > 0.9) {
                shared_ptr<material> sphere_material;

                if (choose_mat < 0.8) {
                    auto albedo = color::random() * color::random();
                    sphere_material = make_shared<lambertian>(albedo);
                    world.add(make_shared<sphere>(center, 0.2, sphere_material));
                } else if (choose_mat < 0.95) {
                    auto albedo = color::random(0.5, 1);
                    auto fuzz = random_double(0, 0.5);
                    sphere_material = make_shared<metal>(albedo, fuzz);
                    world.add(make_shared<sphere>(center, 0.2, sphere_material));
                } else {
                    sphere_material = make_shared<dielectric>(1.5);
                    world.add(make_shared<sphere>(center, 0.2, sphere_material));
                }
            }
        }
    }

    auto material1 = make_shared<dielectric>(1.5);
    world.add(make_shared<sphere>(point3(0, 1, 0), 1.0, material1));

    auto material2 = make_shared<lambertian>(color(0.4, 0.2, 0.1));
    world.add(make_shared<sphere>(point3(-4, 1, 0), 1.0, material2));

    auto material3 = make_shared<metal>(color(0.7, 0.6, 0.5), 0.0);
    world.add(make_shared<sphere>(point3(4, 1, 0), 1.0, material3));

    // Camera setup
    double aspect_ratio = 16.0 / 9.0;
    int image_width = 1200;
    int samples_per_pixel = 20;
    int max_depth = 40;
    double vfov = 20;
    point3 lookfrom = point3(13,2,3);
    point3 lookat = point3(0,0,0);
    vec3 vup = vec3(0,1,0);
    double defocus_angle = 0.6;
    double focus_dist = 10.0;

    int image_height = int(image_width / aspect_ratio);
    image_height = (image_height < 1) ? 1 : image_height;

    std::cerr << "Rendering with Metal GPU (Apple Silicon M3)...\n";
    std::cerr << "Resolution: " << image_width << "x" << image_height << "\n";
    std::cerr << "Samples: " << samples_per_pixel << ", Depth: " << max_depth << "\n\n";

    // Calculate camera parameters manually
    point3 center = lookfrom;
    double theta = degrees_to_radians(vfov);
    double h = std::tan(theta/2);
    double viewport_height = 2 * h * focus_dist;
    double viewport_width = viewport_height * (double(image_width)/image_height);

    vec3 w = unit_vector(lookfrom - lookat);
    vec3 u = unit_vector(cross(vup, w));
    vec3 v = cross(w, u);

    vec3 viewport_u = viewport_width * u;
    vec3 viewport_v = viewport_height * -v;

    vec3 pixel_delta_u = viewport_u / image_width;
    vec3 pixel_delta_v = viewport_v / image_height;

    auto viewport_upper_left = center - (focus_dist * w) - viewport_u/2 - viewport_v/2;
    point3 pixel00_loc = viewport_upper_left + 0.5 * (pixel_delta_u + pixel_delta_v);

    double defocus_radius = focus_dist * std::tan(degrees_to_radians(defocus_angle / 2));
    vec3 defocus_disk_u = u * defocus_radius;
    vec3 defocus_disk_v = v * defocus_radius;

    // Initialize Metal device
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device) {
        std::cerr << "Metal is not supported on this device\n";
        return 1;
    }

    std::cerr << "Using GPU: " << [[device name] UTF8String] << "\n\n";

    // Load and compile shader
    NSError* error = nil;
    NSString* shaderPath = @"raytracer.metal";
    NSString* shaderSource = [NSString stringWithContentsOfFile:shaderPath
                                                        encoding:NSUTF8StringEncoding
                                                           error:&error];

    if (!shaderSource) {
        std::cerr << "Error: Could not find raytracer.metal\n";
        std::cerr << "Make sure raytracer.metal is in the current directory\n";
        return 1;
    }

    id<MTLLibrary> library = [device newLibraryWithSource:shaderSource options:nil error:&error];
    if (!library) {
        std::cerr << "Error compiling shader: " << [[error localizedDescription] UTF8String] << "\n";
        return 1;
    }

    id<MTLFunction> kernel = [library newFunctionWithName:@"raytrace_kernel"];
    if (!kernel) {
        std::cerr << "Error: Could not find raytrace_kernel\n";
        return 1;
    }

    id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:kernel error:&error];
    if (!pipeline) {
        std::cerr << "Error creating pipeline: " << [[error localizedDescription] UTF8String] << "\n";
        return 1;
    }

    // Convert scene to GPU format
    std::vector<GPUSphere> gpu_spheres;

    // Note: This is a simplified extraction - in production, you'd iterate through hittable_list
    // For now, recreate the spheres directly

    // Ground
    GPUSphere ground;
    convert_vec3(point3(0,-1000,0), ground.center);
    ground.radius = 1000;
    ground.material_type = 0; // lambertian
    convert_vec3(color(0.5, 0.5, 0.5), ground.albedo);
    ground.fuzz = 0;
    ground.refraction_index = 0;
    gpu_spheres.push_back(ground);

    // Small spheres
    for (int a = -11; a < 11; a++) {
        for (int b = -11; b < 11; b++) {
            auto choose_mat = random_double();
            point3 center(a + 0.9*random_double(), 0.2, b + 0.9*random_double());

            if ((center - point3(4, 0.2, 0)).length() > 0.9) {
                GPUSphere s;
                convert_vec3(center, s.center);
                s.radius = 0.2;

                if (choose_mat < 0.8) {
                    s.material_type = 0; // lambertian
                    auto albedo = color::random() * color::random();
                    convert_vec3(albedo, s.albedo);
                    s.fuzz = 0;
                    s.refraction_index = 0;
                } else if (choose_mat < 0.95) {
                    s.material_type = 1; // metal
                    auto albedo = color::random(0.5, 1);
                    convert_vec3(albedo, s.albedo);
                    s.fuzz = random_double(0, 0.5);
                    s.refraction_index = 0;
                } else {
                    s.material_type = 2; // dielectric
                    convert_vec3(color(1,1,1), s.albedo);
                    s.fuzz = 0;
                    s.refraction_index = 1.5;
                }
                gpu_spheres.push_back(s);
            }
        }
    }

    // Three large spheres
    GPUSphere s1, s2, s3;

    convert_vec3(point3(0, 1, 0), s1.center);
    s1.radius = 1.0;
    s1.material_type = 2;
    convert_vec3(color(1,1,1), s1.albedo);
    s1.fuzz = 0;
    s1.refraction_index = 1.5;
    gpu_spheres.push_back(s1);

    convert_vec3(point3(-4, 1, 0), s2.center);
    s2.radius = 1.0;
    s2.material_type = 0;
    convert_vec3(color(0.4, 0.2, 0.1), s2.albedo);
    s2.fuzz = 0;
    s2.refraction_index = 0;
    gpu_spheres.push_back(s2);

    convert_vec3(point3(4, 1, 0), s3.center);
    s3.radius = 1.0;
    s3.material_type = 1;
    convert_vec3(color(0.7, 0.6, 0.5), s3.albedo);
    s3.fuzz = 0;
    s3.refraction_index = 0;
    gpu_spheres.push_back(s3);

    std::cerr << "Scene: " << gpu_spheres.size() << " spheres loaded to GPU\n";

    // Setup camera for GPU
    GPUCamera gpu_cam;
    convert_vec3(pixel00_loc, gpu_cam.pixel00_loc);
    convert_vec3(pixel_delta_u, gpu_cam.pixel_delta_u);
    convert_vec3(pixel_delta_v, gpu_cam.pixel_delta_v);
    convert_vec3(center, gpu_cam.center);
    convert_vec3(defocus_disk_u, gpu_cam.defocus_disk_u);
    convert_vec3(defocus_disk_v, gpu_cam.defocus_disk_v);
    gpu_cam.defocus_angle = defocus_angle;

    // Create Metal buffers
    int sphere_count = (int)gpu_spheres.size();

    id<MTLBuffer> sphereBuffer = [device newBufferWithBytes:gpu_spheres.data()
                                                      length:sizeof(GPUSphere) * sphere_count
                                                     options:MTLResourceStorageModeShared];

    id<MTLBuffer> cameraBuffer = [device newBufferWithBytes:&gpu_cam
                                                      length:sizeof(GPUCamera)
                                                     options:MTLResourceStorageModeShared];

    id<MTLBuffer> sphereCountBuffer = [device newBufferWithBytes:&sphere_count
                                                          length:sizeof(int)
                                                         options:MTLResourceStorageModeShared];

    id<MTLBuffer> samplesBuffer = [device newBufferWithBytes:&samples_per_pixel
                                                      length:sizeof(int)
                                                     options:MTLResourceStorageModeShared];

    id<MTLBuffer> depthBuffer = [device newBufferWithBytes:&max_depth
                                                    length:sizeof(int)
                                                   options:MTLResourceStorageModeShared];

    int pixel_count = image_width * image_height;
    id<MTLBuffer> outputBuffer = [device newBufferWithLength:sizeof(float) * 3 * pixel_count
                                                     options:MTLResourceStorageModeShared];

    // Execute on GPU
    id<MTLCommandQueue> queue = [device newCommandQueue];
    id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];

    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:sphereBuffer offset:0 atIndex:0];
    [encoder setBuffer:cameraBuffer offset:0 atIndex:1];
    [encoder setBuffer:sphereCountBuffer offset:0 atIndex:2];
    [encoder setBuffer:samplesBuffer offset:0 atIndex:3];
    [encoder setBuffer:depthBuffer offset:0 atIndex:4];
    [encoder setBuffer:outputBuffer offset:0 atIndex:5];

    MTLSize gridSize = MTLSizeMake(image_width, image_height, 1);
    NSUInteger thread_w = pipeline.threadExecutionWidth;
    NSUInteger thread_h = pipeline.maxTotalThreadsPerThreadgroup / thread_w;
    MTLSize threadgroupSize = MTLSizeMake(thread_w, thread_h, 1);

    [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
    [encoder endEncoding];

    std::cerr << "Dispatching " << pixel_count << " pixels to GPU...\n";

    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];

    std::cerr << "GPU rendering complete!\n\n";

    // Read results and output PPM
    float* output = (float*)[outputBuffer contents];

    std::cout << "P3\n" << image_width << ' ' << image_height << "\n255\n";

    for (int j = 0; j < image_height; j++) {
        std::cerr << "\rWriting scanline: " << j << " / " << image_height << std::flush;
        for (int i = 0; i < image_width; i++) {
            int idx = (j * image_width + i) * 3;

            // Gamma correction
            float r = sqrt(output[idx + 0]);
            float g = sqrt(output[idx + 1]);
            float b = sqrt(output[idx + 2]);

            // Clamp
            r = (r < 0.0f) ? 0.0f : (r > 0.999f) ? 0.999f : r;
            g = (g < 0.0f) ? 0.0f : (g > 0.999f) ? 0.999f : g;
            b = (b < 0.0f) ? 0.0f : (b > 0.999f) ? 0.999f : b;

            int ir = int(256 * r);
            int ig = int(256 * g);
            int ib = int(256 * b);

            std::cout << ir << ' ' << ig << ' ' << ib << '\n';
        }
    }

    std::cerr << "\rDone.                              \n";

    return 0;
}
