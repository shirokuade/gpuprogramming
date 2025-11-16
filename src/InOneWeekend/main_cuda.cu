//==============================================================================================
// Ray Tracing in One Weekend - CUDA GPU Version
// Full GPU acceleration using NVIDIA CUDA
//==============================================================================================

#include <iostream>
#include <cmath>
#include <cstdlib>
#include <cfloat>

// Include the raytracer kernel
#include "raytracer.cu"

// Helper function to check CUDA errors
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - " \
                      << cudaGetErrorString(error) << std::endl; \
            exit(1); \
        } \
    } while(0)

// Helper to convert Vec3 for CPU
struct Vec3CPU {
    float x, y, z;
    Vec3CPU(float x_ = 0, float y_ = 0, float z_ = 0) : x(x_), y(y_), z(z_) {}
};

// Random number generation for scene setup (CPU side)
inline float random_double() {
    return rand() / (RAND_MAX + 1.0);
}

inline float random_double(float min, float max) {
    return min + (max - min) * random_double();
}

int main() {
    // Image settings
    const int image_width = 1200;
    const float aspect_ratio = 16.0f / 9.0f;
    const int image_height = static_cast<int>(image_width / aspect_ratio);
    const int samples_per_pixel = 10;
    const int max_depth = 20;

    // Camera settings
    float vfov = 20.0f;
    Vec3CPU lookfrom(13, 2, 3);
    Vec3CPU lookat(0, 0, 0);
    Vec3CPU vup(0, 1, 0);
    float defocus_angle = 0.6f;
    float focus_dist = 10.0f;

    std::cerr << "Rendering with CUDA GPU...\n";
    std::cerr << "Resolution: " << image_width << "x" << image_height << "\n";
    std::cerr << "Samples: " << samples_per_pixel << ", Depth: " << max_depth << "\n\n";

    // Calculate camera parameters
    float theta = vfov * M_PI / 180.0f;
    float h = tanf(theta / 2.0f);
    float viewport_height = 2.0f * h * focus_dist;
    float viewport_width = viewport_height * (float(image_width) / image_height);

    Vec3CPU w_cpu = Vec3CPU(lookfrom.x - lookat.x, lookfrom.y - lookat.y, lookfrom.z - lookat.z);
    float w_len = sqrtf(w_cpu.x*w_cpu.x + w_cpu.y*w_cpu.y + w_cpu.z*w_cpu.z);
    w_cpu.x /= w_len; w_cpu.y /= w_len; w_cpu.z /= w_len;

    Vec3CPU u_cpu = Vec3CPU(
        vup.y * w_cpu.z - vup.z * w_cpu.y,
        vup.z * w_cpu.x - vup.x * w_cpu.z,
        vup.x * w_cpu.y - vup.y * w_cpu.x
    );
    float u_len = sqrtf(u_cpu.x*u_cpu.x + u_cpu.y*u_cpu.y + u_cpu.z*u_cpu.z);
    u_cpu.x /= u_len; u_cpu.y /= u_len; u_cpu.z /= u_len;

    Vec3CPU v_cpu = Vec3CPU(
        w_cpu.y * u_cpu.z - w_cpu.z * u_cpu.y,
        w_cpu.z * u_cpu.x - w_cpu.x * u_cpu.z,
        w_cpu.x * u_cpu.y - w_cpu.y * u_cpu.x
    );

    Vec3CPU viewport_u(viewport_width * u_cpu.x, viewport_width * u_cpu.y, viewport_width * u_cpu.z);
    Vec3CPU viewport_v(-viewport_height * v_cpu.x, -viewport_height * v_cpu.y, -viewport_height * v_cpu.z);

    Vec3CPU pixel_delta_u(viewport_u.x / image_width, viewport_u.y / image_width, viewport_u.z / image_width);
    Vec3CPU pixel_delta_v(viewport_v.x / image_height, viewport_v.y / image_height, viewport_v.z / image_height);

    Vec3CPU viewport_upper_left(
        lookfrom.x - (focus_dist * w_cpu.x) - viewport_u.x/2 - viewport_v.x/2,
        lookfrom.y - (focus_dist * w_cpu.y) - viewport_u.y/2 - viewport_v.y/2,
        lookfrom.z - (focus_dist * w_cpu.z) - viewport_u.z/2 - viewport_v.z/2
    );

    Vec3CPU pixel00_loc(
        viewport_upper_left.x + 0.5f * (pixel_delta_u.x + pixel_delta_v.x),
        viewport_upper_left.y + 0.5f * (pixel_delta_u.y + pixel_delta_v.y),
        viewport_upper_left.z + 0.5f * (pixel_delta_u.z + pixel_delta_v.z)
    );

    float defocus_radius = focus_dist * tanf((defocus_angle * M_PI / 180.0f) / 2.0f);
    Vec3CPU defocus_disk_u(u_cpu.x * defocus_radius, u_cpu.y * defocus_radius, u_cpu.z * defocus_radius);
    Vec3CPU defocus_disk_v(v_cpu.x * defocus_radius, v_cpu.y * defocus_radius, v_cpu.z * defocus_radius);

    // Setup camera on host
    Camera h_camera;
    h_camera.pixel00_loc = Vec3(pixel00_loc.x, pixel00_loc.y, pixel00_loc.z);
    h_camera.pixel_delta_u = Vec3(pixel_delta_u.x, pixel_delta_u.y, pixel_delta_u.z);
    h_camera.pixel_delta_v = Vec3(pixel_delta_v.x, pixel_delta_v.y, pixel_delta_v.z);
    h_camera.center = Vec3(lookfrom.x, lookfrom.y, lookfrom.z);
    h_camera.defocus_disk_u = Vec3(defocus_disk_u.x, defocus_disk_u.y, defocus_disk_u.z);
    h_camera.defocus_disk_v = Vec3(defocus_disk_v.x, defocus_disk_v.y, defocus_disk_v.z);
    h_camera.defocus_angle = defocus_angle;

    // Create scene
    std::vector<Sphere> h_spheres;

    // Ground
    Sphere ground;
    ground.center = Vec3(0, -1000, 0);
    ground.radius = 1000;
    ground.material_type = 0; // lambertian
    ground.albedo = Vec3(0.5f, 0.5f, 0.5f);
    ground.fuzz = 0;
    ground.refraction_index = 0;
    h_spheres.push_back(ground);

    // Random small spheres
    srand(1234);  // Fixed seed for reproducibility
    for (int a = -11; a < 11; a++) {
        for (int b = -11; b < 11; b++) {
            float choose_mat = random_double();
            Vec3CPU center(a + 0.9f*random_double(), 0.2f, b + 0.9f*random_double());
            Vec3CPU check(center.x - 4, center.y - 0.2f, center.z);
            float dist = sqrtf(check.x*check.x + check.y*check.y + check.z*check.z);

            if (dist > 0.9f) {
                Sphere sphere;
                sphere.center = Vec3(center.x, center.y, center.z);
                sphere.radius = 0.2f;

                if (choose_mat < 0.8f) {
                    // Diffuse
                    Vec3CPU albedo(random_double() * random_double(),
                                   random_double() * random_double(),
                                   random_double() * random_double());
                    sphere.material_type = 0;
                    sphere.albedo = Vec3(albedo.x, albedo.y, albedo.z);
                    sphere.fuzz = 0;
                    sphere.refraction_index = 0;
                } else if (choose_mat < 0.95f) {
                    // Metal
                    Vec3CPU albedo(random_double(0.5f, 1.0f),
                                   random_double(0.5f, 1.0f),
                                   random_double(0.5f, 1.0f));
                    sphere.material_type = 1;
                    sphere.albedo = Vec3(albedo.x, albedo.y, albedo.z);
                    sphere.fuzz = random_double(0, 0.5f);
                    sphere.refraction_index = 0;
                } else {
                    // Glass
                    sphere.material_type = 2;
                    sphere.albedo = Vec3(1.0f, 1.0f, 1.0f);
                    sphere.fuzz = 0;
                    sphere.refraction_index = 1.5f;
                }
                h_spheres.push_back(sphere);
            }
        }
    }

    // Three large spheres
    Sphere s1;
    s1.center = Vec3(0, 1, 0);
    s1.radius = 1.0f;
    s1.material_type = 2; // glass
    s1.albedo = Vec3(1.0f, 1.0f, 1.0f);
    s1.fuzz = 0;
    s1.refraction_index = 1.5f;
    h_spheres.push_back(s1);

    Sphere s2;
    s2.center = Vec3(-4, 1, 0);
    s2.radius = 1.0f;
    s2.material_type = 0; // diffuse
    s2.albedo = Vec3(0.4f, 0.2f, 0.1f);
    s2.fuzz = 0;
    s2.refraction_index = 0;
    h_spheres.push_back(s2);

    Sphere s3;
    s3.center = Vec3(4, 1, 0);
    s3.radius = 1.0f;
    s3.material_type = 1; // metal
    s3.albedo = Vec3(0.7f, 0.6f, 0.5f);
    s3.fuzz = 0.0f;
    s3.refraction_index = 0;
    h_spheres.push_back(s3);

    int sphere_count = h_spheres.size();
    std::cerr << "Scene: " << sphere_count << " spheres\n";

    // Allocate device memory
    Sphere* d_spheres;
    Vec3* d_output;
    curandState* d_rand_state;

    int pixel_count = image_width * image_height;

    CUDA_CHECK(cudaMalloc(&d_spheres, sphere_count * sizeof(Sphere)));
    CUDA_CHECK(cudaMalloc(&d_output, pixel_count * sizeof(Vec3)));
    CUDA_CHECK(cudaMalloc(&d_rand_state, pixel_count * sizeof(curandState)));

    // Copy scene to device
    CUDA_CHECK(cudaMemcpy(d_spheres, h_spheres.data(), sphere_count * sizeof(Sphere), cudaMemcpyHostToDevice));

    // Initialize random states
    dim3 block_size(16, 16);
    dim3 grid_size((image_width + block_size.x - 1) / block_size.x,
                   (image_height + block_size.y - 1) / block_size.y);

    std::cerr << "Initializing random states...\n";
    init_rand<<<grid_size, block_size>>>(d_rand_state, image_width, image_height, time(0));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Launch raytracing kernel
    std::cerr << "Launching CUDA raytracing kernel...\n";
    std::cerr << "Grid: " << grid_size.x << "x" << grid_size.y << ", Block: " << block_size.x << "x" << block_size.y << "\n";
    std::cerr << "Total threads: " << pixel_count << "\n\n";

    raytrace_kernel<<<grid_size, block_size>>>(
        d_output,
        d_spheres,
        h_camera,
        sphere_count,
        samples_per_pixel,
        max_depth,
        image_width,
        image_height,
        d_rand_state
    );

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::cerr << "GPU rendering complete!\n\n";

    // Copy result back to host
    Vec3* h_output = new Vec3[pixel_count];
    CUDA_CHECK(cudaMemcpy(h_output, d_output, pixel_count * sizeof(Vec3), cudaMemcpyDeviceToHost));

    // Output PPM
    std::cout << "P3\n" << image_width << ' ' << image_height << "\n255\n";

    for (int j = 0; j < image_height; j++) {
        std::cerr << "\rWriting scanline: " << j << " / " << image_height << std::flush;
        for (int i = 0; i < image_width; i++) {
            int idx = j * image_width + i;

            // Gamma correction
            float r = sqrtf(h_output[idx].x);
            float g = sqrtf(h_output[idx].y);
            float b = sqrtf(h_output[idx].z);

            // Clamp
            r = (r < 0.0f) ? 0.0f : (r > 0.999f) ? 0.999f : r;
            g = (g < 0.0f) ? 0.0f : (g > 0.999f) ? 0.999f : g;
            b = (b < 0.0f) ? 0.0f : (b > 0.999f) ? 0.999f : b;

            int ir = static_cast<int>(256 * r);
            int ig = static_cast<int>(256 * g);
            int ib = static_cast<int>(256 * b);

            std::cout << ir << ' ' << ig << ' ' << ib << '\n';
        }
    }

    std::cerr << "\rDone.                              \n";

    // Cleanup
    delete[] h_output;
    CUDA_CHECK(cudaFree(d_spheres));
    CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaFree(d_rand_state));

    return 0;
}
