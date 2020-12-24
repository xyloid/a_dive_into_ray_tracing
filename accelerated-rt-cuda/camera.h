#ifndef CAMERA_H
#define CAMERA_H

#include "rtweekend.h"

class camera {
public:
  __device__ camera(vec3 lookfrom, vec3 lookat, vec3 vup, float vfov,
                    float aspect) {
    // float aspect_ratio = 16.0 / 9.0;
    // float viewport_height = 2.0;
    // float viewport_width = aspect_ratio * viewport_height;
    // float focal_length = 1.0;

    // origin = point3(0, 0, 0);
    // horizontal = vec3(viewport_width, 0.0, 0.0);
    // vertical = vec3(0.0, viewport_height, 0.0);
    // lower_left_corner =
    //     origin - horizontal / 2 - vertical / 2 - vec3(0, 0, focal_length);

    // vfov is top to bottom in degrees.
    vec3 u, v, w;
    float theta = vfov * M_PI / 180;
    float half_height = tan(theta / 2);
    float half_width = aspect * half_height;
    origin = lookfrom;

    w = unit_vector(lookfrom - lookat);
    u = unit_vector(cross(vup, w));
    v = cross(w, u);

    lower_left_corner = origin - half_width * u - half_height * v - w;
    horizontal = 2 * half_width * u;
    vertical = 2 * half_height * v;
  }

  __device__ ray get_ray(float u, float v) const {
    return ray(origin,
               lower_left_corner + u * horizontal + v * vertical - origin);
  }

private:
  point3 origin;
  point3 lower_left_corner;
  vec3 horizontal;
  vec3 vertical;
};
#endif