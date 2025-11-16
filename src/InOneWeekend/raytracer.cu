//==============================================================================================
// CUDA GPU Raytracer - Optimized for NVIDIA GPUs
// Converted from CPU version for massive parallel processing
//==============================================================================================

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <float.h>

// CUDA-compatible Vec3 structure
struct Vec3 {
    float x, y, z;

    __host__ __device__ Vec3() : x(0), y(0), z(0) {}
    __host__ __device__ Vec3(float x_, float y_, float z_) : x(x_), y(y_), z(z_) {}

    __host__ __device__ float length_squared() const {
        return x*x + y*y + z*z;
    }

    __host__ __device__ float length() const {
        return sqrtf(length_squared());
    }
};

// Vec3 operators
__host__ __device__ Vec3 operator+(const Vec3& a, const Vec3& b) {
    return Vec3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__host__ __device__ Vec3 operator-(const Vec3& a, const Vec3& b) {
    return Vec3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__host__ __device__ Vec3 operator*(const Vec3& a, const Vec3& b) {
    return Vec3(a.x * b.x, a.y * b.y, a.z * b.z);
}

__host__ __device__ Vec3 operator*(float t, const Vec3& v) {
    return Vec3(t * v.x, t * v.y, t * v.z);
}

__host__ __device__ Vec3 operator*(const Vec3& v, float t) {
    return t * v;
}

__host__ __device__ Vec3 operator/(const Vec3& v, float t) {
    return (1.0f / t) * v;
}

__host__ __device__ Vec3 operator-(const Vec3& v) {
    return Vec3(-v.x, -v.y, -v.z);
}

__host__ __device__ float dot(const Vec3& a, const Vec3& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__host__ __device__ Vec3 unit_vector(const Vec3& v) {
    return v / v.length();
}

__host__ __device__ Vec3 reflect(const Vec3& v, const Vec3& n) {
    return v - 2.0f * dot(v, n) * n;
}

__host__ __device__ Vec3 refract(const Vec3& uv, const Vec3& n, float etai_over_etat) {
    float cos_theta = fminf(dot(-uv, n), 1.0f);
    Vec3 r_out_perp = etai_over_etat * (uv + cos_theta * n);
    Vec3 r_out_parallel = -sqrtf(fabsf(1.0f - r_out_perp.length_squared())) * n;
    return r_out_perp + r_out_parallel;
}

__host__ __device__ bool near_zero(const Vec3& v) {
    const float s = 1e-8f;
    return (fabsf(v.x) < s) && (fabsf(v.y) < s) && (fabsf(v.z) < s);
}

// Random number generation using cuRAND
__device__ Vec3 random_in_unit_sphere(curandState* rand_state) {
    Vec3 p;
    do {
        p = 2.0f * Vec3(curand_uniform(rand_state), curand_uniform(rand_state), curand_uniform(rand_state)) - Vec3(1, 1, 1);
    } while (p.length_squared() >= 1.0f);
    return p;
}

__device__ Vec3 random_unit_vector(curandState* rand_state) {
    return unit_vector(random_in_unit_sphere(rand_state));
}

__device__ Vec3 random_in_unit_disk(curandState* rand_state) {
    Vec3 p;
    do {
        p = 2.0f * Vec3(curand_uniform(rand_state), curand_uniform(rand_state), 0) - Vec3(1, 1, 0);
    } while (dot(p, p) >= 1.0f);
    return p;
}

// Ray structure
struct Ray {
    Vec3 origin;
    Vec3 direction;
};

// Sphere structure
struct Sphere {
    Vec3 center;
    float radius;
    int material_type;  // 0=lambertian, 1=metal, 2=dielectric
    Vec3 albedo;
    float fuzz;
    float refraction_index;
};

// Hit record
struct HitRecord {
    Vec3 p;
    Vec3 normal;
    float t;
    bool front_face;
    int material_type;
    Vec3 albedo;
    float fuzz;
    float refraction_index;
};

// Camera structure
struct Camera {
    Vec3 pixel00_loc;
    Vec3 pixel_delta_u;
    Vec3 pixel_delta_v;
    Vec3 center;
    Vec3 defocus_disk_u;
    Vec3 defocus_disk_v;
    float defocus_angle;
};

// Ray-sphere intersection
__device__ bool hit_sphere(const Sphere& sphere, const Ray& r, float ray_tmin, float ray_tmax, HitRecord& rec) {
    Vec3 oc = sphere.center - r.origin;
    float a = dot(r.direction, r.direction);
    float h = dot(r.direction, oc);
    float c = dot(oc, oc) - sphere.radius * sphere.radius;

    float discriminant = h*h - a*c;
    if (discriminant < 0)
        return false;

    float sqrtd = sqrtf(discriminant);
    float root = (h - sqrtd) / a;

    if (root <= ray_tmin || ray_tmax <= root) {
        root = (h + sqrtd) / a;
        if (root <= ray_tmin || ray_tmax <= root)
            return false;
    }

    rec.t = root;
    rec.p = r.origin + root * r.direction;
    Vec3 outward_normal = (rec.p - sphere.center) / sphere.radius;
    rec.front_face = dot(r.direction, outward_normal) < 0;
    rec.normal = rec.front_face ? outward_normal : -outward_normal;
    rec.material_type = sphere.material_type;
    rec.albedo = sphere.albedo;
    rec.fuzz = sphere.fuzz;
    rec.refraction_index = sphere.refraction_index;

    return true;
}

// Hit all spheres
__device__ bool hit_world(const Sphere* spheres, int sphere_count, const Ray& r, float ray_tmin, float ray_tmax, HitRecord& rec) {
    HitRecord temp_rec;
    bool hit_anything = false;
    float closest_so_far = ray_tmax;

    for (int i = 0; i < sphere_count; i++) {
        if (hit_sphere(spheres[i], r, ray_tmin, closest_so_far, temp_rec)) {
            hit_anything = true;
            closest_so_far = temp_rec.t;
            rec = temp_rec;
        }
    }

    return hit_anything;
}

// Schlick's approximation
__device__ float reflectance(float cosine, float ref_idx) {
    float r0 = (1.0f - ref_idx) / (1.0f + ref_idx);
    r0 = r0 * r0;
    return r0 + (1.0f - r0) * powf((1.0f - cosine), 5.0f);
}

// Material scattering
__device__ bool scatter(const HitRecord& rec, const Ray& r_in, Vec3& attenuation, Ray& scattered, curandState* rand_state) {
    if (rec.material_type == 0) {  // Lambertian
        Vec3 scatter_direction = rec.normal + random_unit_vector(rand_state);
        if (near_zero(scatter_direction))
            scatter_direction = rec.normal;
        scattered.origin = rec.p;
        scattered.direction = scatter_direction;
        attenuation = rec.albedo;
        return true;
    }
    else if (rec.material_type == 1) {  // Metal
        Vec3 reflected = reflect(unit_vector(r_in.direction), rec.normal);
        reflected = unit_vector(reflected) + (rec.fuzz * random_unit_vector(rand_state));
        scattered.origin = rec.p;
        scattered.direction = reflected;
        attenuation = rec.albedo;
        return (dot(scattered.direction, rec.normal) > 0);
    }
    else if (rec.material_type == 2) {  // Dielectric
        attenuation = Vec3(1.0f, 1.0f, 1.0f);
        float ri = rec.front_face ? (1.0f / rec.refraction_index) : rec.refraction_index;

        Vec3 unit_direction = unit_vector(r_in.direction);
        float cos_theta = fminf(dot(-unit_direction, rec.normal), 1.0f);
        float sin_theta = sqrtf(1.0f - cos_theta * cos_theta);

        bool cannot_refract = ri * sin_theta > 1.0f;
        Vec3 direction;

        if (cannot_refract || reflectance(cos_theta, ri) > curand_uniform(rand_state))
            direction = reflect(unit_direction, rec.normal);
        else
            direction = refract(unit_direction, rec.normal, ri);

        scattered.origin = rec.p;
        scattered.direction = direction;
        return true;
    }

    return false;
}

// Ray color calculation
__device__ Vec3 ray_color(const Ray& r, const Sphere* spheres, int sphere_count, int max_depth, curandState* rand_state) {
    Ray current_ray = r;
    Vec3 accumulated_color = Vec3(1.0f, 1.0f, 1.0f);

    for (int depth = 0; depth < max_depth; depth++) {
        HitRecord rec;

        if (hit_world(spheres, sphere_count, current_ray, 0.001f, FLT_MAX, rec)) {
            Ray scattered;
            Vec3 attenuation;
            if (scatter(rec, current_ray, attenuation, scattered, rand_state)) {
                accumulated_color = accumulated_color * attenuation;
                current_ray = scattered;
            } else {
                return Vec3(0, 0, 0);
            }
        } else {
            // Sky gradient
            Vec3 unit_direction = unit_vector(current_ray.direction);
            float a = 0.5f * (unit_direction.y + 1.0f);
            Vec3 sky = (1.0f - a) * Vec3(1.0f, 1.0f, 1.0f) + a * Vec3(0.5f, 0.7f, 1.0f);
            return accumulated_color * sky;
        }
    }

    return Vec3(0, 0, 0);  // Exceeded depth
}

// Initialize cuRAND state for each thread
__global__ void init_rand(curandState* rand_state, int width, int height, unsigned long long seed) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if (i >= width || j >= height) return;

    int pixel_index = j * width + i;
    curand_init(seed + pixel_index, 0, 0, &rand_state[pixel_index]);
}

// Main raytracing kernel - one thread per pixel
__global__ void raytrace_kernel(
    Vec3* output,
    const Sphere* spheres,
    const Camera camera,
    int sphere_count,
    int samples_per_pixel,
    int max_depth,
    int width,
    int height,
    curandState* rand_state
) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if (i >= width || j >= height) return;

    int pixel_index = j * width + i;
    curandState local_rand_state = rand_state[pixel_index];

    Vec3 pixel_color(0, 0, 0);

    // Multi-sampling
    for (int sample = 0; sample < samples_per_pixel; sample++) {
        // Random offset within pixel
        float offset_x = curand_uniform(&local_rand_state) - 0.5f;
        float offset_y = curand_uniform(&local_rand_state) - 0.5f;

        Vec3 pixel_sample = camera.pixel00_loc
                          + (float(i) + offset_x) * camera.pixel_delta_u
                          + (float(j) + offset_y) * camera.pixel_delta_v;

        Vec3 ray_origin;
        if (camera.defocus_angle <= 0) {
            ray_origin = camera.center;
        } else {
            Vec3 p = random_in_unit_disk(&local_rand_state);
            ray_origin = camera.center + (p.x * camera.defocus_disk_u) + (p.y * camera.defocus_disk_v);
        }

        Ray r;
        r.origin = ray_origin;
        r.direction = pixel_sample - ray_origin;

        pixel_color = pixel_color + ray_color(r, spheres, sphere_count, max_depth, &local_rand_state);
    }

    // Save updated random state
    rand_state[pixel_index] = local_rand_state;

    // Average and write output
    float scale = 1.0f / float(samples_per_pixel);
    output[pixel_index] = pixel_color * scale;
}
