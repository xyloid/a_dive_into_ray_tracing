#ifndef CAMERA_H
#define CAMERA_H

#include "rtweekend.h"

#include <curand_kernel.h>

__device__ vec3 random_in_unit_disk(curandState *local_rand_state) {
  vec3 p;
  do {
    p = 2.0f * vec3(curand_uniform(local_rand_state),
                    curand_uniform(local_rand_state), 0) -
        vec3(1, 1, 0);
  } while (dot(p, p) >= 1.0f);
  return p;
}

// __device__ float random_float(float min, float max,
//                               curandState *local_rand_state) {
//   return curand_uniform(local_rand_state) * (max - min) + min;
// }

class camera {
public:
  __device__ camera(vec3 lookfrom, vec3 lookat, vec3 vup, float vfov,
                    float aspect, float aperture, float focal_dist,
                    float _time0 = 0.0, float _time1 = 0.0) {
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

    float theta = vfov * M_PI / 180;
    float half_height = tan(theta / 2);
    float half_width = aspect * half_height;
    origin = lookfrom;

    w = unit_vector(lookfrom - lookat);
    // w = unit_vector(lookat - lookfrom);
    u = unit_vector(cross(vup, w));
    v = cross(w, u);

    // lower_left_corner = origin - half_width * u - half_height * v - w;
    // horizontal = 2 * half_width * u;
    // vertical = 2 * half_height * v;
    horizontal = focal_dist * 2.0f * half_width * u;
    vertical = focal_dist * 2.0f * half_height * v;
    lower_left_corner =
        origin - horizontal / 2.0f - vertical / 2.0f - focal_dist * w;

    lens_radius = aperture / 2.0f;

    time0 = _time0;
    time1 = _time1;
  }

  __device__ ray get_ray(float s, float t,
                         curandState *local_rand_state) const {
    /**
     * The below line is the difference
     */
    vec3 rd = lens_radius * random_in_unit_disk(local_rand_state);
    // vec3 rd = random_in_unit_disk(local_rand_state);
    // vec3 rd2(lens_radius * rd[0], lens_radius * rd[1], lens_radius * rd[2]);

    vec3 offset = u * rd.x() + v * rd.y();
    // printf("offset %f %f %f\n", offset.x(), offset.y(), offset.z());
    // direction pointing to negative direction
    return ray(origin + offset,
               lower_left_corner + s * horizontal + t * vertical - origin -
                   offset,
               random_float(time0, time1, local_rand_state));
    // return ray();
  }

private:
  point3 origin;
  point3 lower_left_corner;
  vec3 horizontal;
  vec3 vertical;
  vec3 u, v, w;
  float lens_radius;
  float time0;
  float time1; // shutter open/close time
};
#endif