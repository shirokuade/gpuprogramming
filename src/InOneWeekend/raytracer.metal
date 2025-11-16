//==============================================================================================
// Metal GPU Raytracer - Optimized for Apple Silicon M3
// Converted from CPU version for GPU acceleration
//==============================================================================================

#include <metal_stdlib>
using namespace metal;

// Constants
constant float PI = 3.1415926535897932385;
constant float INFINITY_VAL = 1e10;

// Data structures matching CPU version
struct Vec3 {
    float x, y, z;

    Vec3(float x_ = 0, float y_ = 0, float z_ = 0) : x(x_), y(y_), z(z_) {}

    float length_squared() const {
        return x*x + y*y + z*z;
    }

    float length() const {
        return sqrt(length_squared());
    }
};

struct Ray {
    Vec3 origin;
    Vec3 direction;
};

struct Sphere {
    Vec3 center;
    float radius;
    int material_type;  // 0=lambertian, 1=metal, 2=dielectric
    Vec3 albedo;        // color/albedo
    float fuzz;         // for metal
    float refraction_index; // for dielectric
};

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

struct Camera {
    Vec3 pixel00_loc;
    Vec3 pixel_delta_u;
    Vec3 pixel_delta_v;
    Vec3 center;
    Vec3 defocus_disk_u;
    Vec3 defocus_disk_v;
    float defocus_angle;
};

// Random number generation (using thread ID as seed)
float random_float(thread uint& seed) {
    seed = seed * 747796405u + 2891336453u;
    uint result = ((seed >> ((seed >> 28u) + 4u)) ^ seed) * 277803737u;
    result = (result >> 22u) ^ result;
    return result / 4294967295.0f;
}

float random_float_range(float min, float max, thread uint& seed) {
    return min + (max - min) * random_float(seed);
}

// Vec3 operations
Vec3 operator+(Vec3 a, Vec3 b) {
    return Vec3(a.x + b.x, a.y + b.y, a.z + b.z);
}

Vec3 operator-(Vec3 a, Vec3 b) {
    return Vec3(a.x - b.x, a.y - b.y, a.z - b.z);
}

Vec3 operator*(Vec3 a, Vec3 b) {
    return Vec3(a.x * b.x, a.y * b.y, a.z * b.z);
}

Vec3 operator*(float t, Vec3 v) {
    return Vec3(t * v.x, t * v.y, t * v.z);
}

Vec3 operator*(Vec3 v, float t) {
    return t * v;
}

Vec3 operator/(Vec3 v, float t) {
    return (1.0f / t) * v;
}

Vec3 operator-(Vec3 v) {
    return Vec3(-v.x, -v.y, -v.z);
}

float dot(Vec3 a, Vec3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

Vec3 unit_vector(Vec3 v) {
    return v / v.length();
}

Vec3 reflect(Vec3 v, Vec3 n) {
    return v - 2.0f * dot(v, n) * n;
}

Vec3 refract(Vec3 uv, Vec3 n, float etai_over_etat) {
    float cos_theta = fmin(dot(-uv, n), 1.0f);
    Vec3 r_out_perp = etai_over_etat * (uv + cos_theta * n);
    Vec3 r_out_parallel = -sqrt(fabs(1.0f - r_out_perp.length_squared())) * n;
    return r_out_perp + r_out_parallel;
}

Vec3 random_in_unit_sphere(thread uint& seed) {
    while (true) {
        Vec3 p = Vec3(random_float_range(-1, 1, seed),
                      random_float_range(-1, 1, seed),
                      random_float_range(-1, 1, seed));
        if (p.length_squared() < 1.0f)
            return p;
    }
}

Vec3 random_unit_vector(thread uint& seed) {
    while (true) {
        Vec3 p = Vec3(random_float_range(-1, 1, seed),
                      random_float_range(-1, 1, seed),
                      random_float_range(-1, 1, seed));
        float lensq = p.length_squared();
        if (1e-160 < lensq && lensq <= 1.0f)
            return p / sqrt(lensq);
    }
}

Vec3 random_in_unit_disk(thread uint& seed) {
    while (true) {
        Vec3 p = Vec3(random_float_range(-1, 1, seed),
                      random_float_range(-1, 1, seed),
                      0);
        if (p.length_squared() < 1.0f)
            return p;
    }
}

bool near_zero(Vec3 v) {
    float s = 1e-8;
    return (fabs(v.x) < s) && (fabs(v.y) < s) && (fabs(v.z) < s);
}

// Ray-sphere intersection
bool hit_sphere(constant Sphere& sphere, Ray r, float ray_tmin, float ray_tmax, thread HitRecord& rec) {
    Vec3 oc = sphere.center - r.origin;
    float a = r.direction.length_squared();
    float h = dot(r.direction, oc);
    float c = oc.length_squared() - sphere.radius * sphere.radius;

    float discriminant = h*h - a*c;
    if (discriminant < 0)
        return false;

    float sqrtd = sqrt(discriminant);
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

// Hit all spheres in scene
bool hit_world(constant Sphere* spheres, int sphere_count, Ray r, float ray_tmin, float ray_tmax, thread HitRecord& rec) {
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

// Schlick's approximation for reflectance
float reflectance(float cosine, float ref_idx) {
    float r0 = (1.0f - ref_idx) / (1.0f + ref_idx);
    r0 = r0 * r0;
    return r0 + (1.0f - r0) * pow((1.0f - cosine), 5.0f);
}

// Material scattering
bool scatter(thread HitRecord& rec, thread Ray& r_in, thread Vec3& attenuation, thread Ray& scattered, thread uint& seed) {
    if (rec.material_type == 0) {  // Lambertian
        Vec3 scatter_direction = rec.normal + random_unit_vector(seed);
        if (near_zero(scatter_direction))
            scatter_direction = rec.normal;
        scattered.origin = rec.p;
        scattered.direction = scatter_direction;
        attenuation = rec.albedo;
        return true;
    }
    else if (rec.material_type == 1) {  // Metal
        Vec3 reflected = reflect(unit_vector(r_in.direction), rec.normal);
        reflected = unit_vector(reflected) + (rec.fuzz * random_unit_vector(seed));
        scattered.origin = rec.p;
        scattered.direction = reflected;
        attenuation = rec.albedo;
        return (dot(scattered.direction, rec.normal) > 0);
    }
    else if (rec.material_type == 2) {  // Dielectric
        attenuation = Vec3(1.0f, 1.0f, 1.0f);
        float ri = rec.front_face ? (1.0f / rec.refraction_index) : rec.refraction_index;

        Vec3 unit_direction = unit_vector(r_in.direction);
        float cos_theta = fmin(dot(-unit_direction, rec.normal), 1.0f);
        float sin_theta = sqrt(1.0f - cos_theta * cos_theta);

        bool cannot_refract = ri * sin_theta > 1.0f;
        Vec3 direction;

        if (cannot_refract || reflectance(cos_theta, ri) > random_float(seed))
            direction = reflect(unit_direction, rec.normal);
        else
            direction = refract(unit_direction, rec.normal, ri);

        scattered.origin = rec.p;
        scattered.direction = direction;
        return true;
    }

    return false;
}

// Ray color calculation with recursive bouncing
Vec3 ray_color(Ray r, constant Sphere* spheres, int sphere_count, int max_depth, thread uint& seed) {
    Vec3 accumulated_color = Vec3(1.0f, 1.0f, 1.0f);
    Ray current_ray = r;

    for (int depth = 0; depth < max_depth; depth++) {
        HitRecord rec;

        if (hit_world(spheres, sphere_count, current_ray, 0.001f, INFINITY_VAL, rec)) {
            Ray scattered;
            Vec3 attenuation;
            if (scatter(rec, current_ray, attenuation, scattered, seed)) {
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

// Main kernel - one thread per pixel
kernel void raytrace_kernel(
    constant Sphere* spheres [[buffer(0)]],
    constant Camera& camera [[buffer(1)]],
    constant int& sphere_count [[buffer(2)]],
    constant int& samples_per_pixel [[buffer(3)]],
    constant int& max_depth [[buffer(4)]],
    device float3* output [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]],
    uint2 grid_size [[threads_per_grid]]
) {
    uint pixel_index = gid.y * grid_size.x + gid.x;
    uint seed = pixel_index * 719393;  // Unique seed per pixel

    Vec3 pixel_color = Vec3(0, 0, 0);

    // Multi-sampling
    for (int sample = 0; sample < samples_per_pixel; sample++) {
        // Random offset within pixel
        float offset_x = random_float(seed) - 0.5f;
        float offset_y = random_float(seed) - 0.5f;

        Vec3 pixel_sample = camera.pixel00_loc
                          + (float(gid.x) + offset_x) * camera.pixel_delta_u
                          + (float(gid.y) + offset_y) * camera.pixel_delta_v;

        Vec3 ray_origin;
        if (camera.defocus_angle <= 0) {
            ray_origin = camera.center;
        } else {
            Vec3 p = random_in_unit_disk(seed);
            ray_origin = camera.center + (p.x * camera.defocus_disk_u) + (p.y * camera.defocus_disk_v);
        }

        Ray r;
        r.origin = ray_origin;
        r.direction = pixel_sample - ray_origin;

        pixel_color = pixel_color + ray_color(r, spheres, sphere_count, max_depth, seed);
    }

    // Average and write output
    float scale = 1.0f / float(samples_per_pixel);
    output[pixel_index] = float3(pixel_color.x * scale, pixel_color.y * scale, pixel_color.z * scale);
}
