#ifndef CAMERA_GPU_H
#define CAMERA_GPU_H

#include "camera.h"

// Extended camera class with public access for GPU rendering
class camera_gpu : public camera {
public:
    // Make initialization public for GPU access
    void public_initialize() {
        initialize();
    }

    // Public accessors for GPU rendering
    point3 get_pixel00_loc() const { return pixel00_loc; }
    vec3 get_pixel_delta_u() const { return pixel_delta_u; }
    vec3 get_pixel_delta_v() const { return pixel_delta_v; }
    point3 get_center() const { return center; }
    vec3 get_defocus_disk_u() const { return defocus_disk_u; }
    vec3 get_defocus_disk_v() const { return defocus_disk_v; }

private:
    // Access to parent's private members through protected/public interface
    using camera::initialize;
    using camera::pixel00_loc;
    using camera::pixel_delta_u;
    using camera::pixel_delta_v;
    using camera::center;
    using camera::defocus_disk_u;
    using camera::defocus_disk_v;
};

#endif
