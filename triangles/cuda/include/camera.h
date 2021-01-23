#ifndef CAMERA_H
#define CAMERA_H

#include "rtweekend.h"

#include <curand_kernel.h>

__device__ vec3 random_in_unit_disk(curandState *local_rand_state) {
  vec3 p;
  do {
    p = 2.0 * vec3(curand_uniform(local_rand_state),
                    curand_uniform(local_rand_state), 0) -
        vec3(1, 1, 0);
  } while (dot(p, p) >= 1.0);
  return p;
}

// __device__ double random_double(double min, double max,
//                               curandState *local_rand_state) {
//   return curand_uniform(local_rand_state) * (max - min) + min;
// }

class camera {
public:
  __device__ camera(vec3 lookfrom, vec3 lookat, vec3 vup, double vfov,
                    double aspect, double aperture, double focal_dist,
                    double _time0 = 0.0, double _time1 = 0.0) {
    // double aspect_ratio = 16.0 / 9.0;
    // double viewport_height = 2.0;
    // double viewport_width = aspect_ratio * viewport_height;
    // double focal_length = 1.0;

    // origin = point3(0, 0, 0);
    // horizontal = vec3(viewport_width, 0.0, 0.0);
    // vertical = vec3(0.0, viewport_height, 0.0);
    // lower_left_corner =
    //     origin - horizontal / 2 - vertical / 2 - vec3(0, 0, focal_length);

    // vfov is top to bottom in degrees.

    double theta = vfov * M_PI / 180;
    double half_height = tan(theta / 2);
    double half_width = aspect * half_height;
    origin = lookfrom;

    w = unit_vector(lookfrom - lookat);
    // w = unit_vector(lookat - lookfrom);
    u = unit_vector(cross(vup, w));
    v = cross(w, u);

    // lower_left_corner = origin - half_width * u - half_height * v - w;
    // horizontal = 2 * half_width * u;
    // vertical = 2 * half_height * v;
    horizontal = focal_dist * 2.0 * half_width * u;
    vertical = focal_dist * 2.0 * half_height * v;
    lower_left_corner =
        origin - horizontal / 2.0 - vertical / 2.0 - focal_dist * w;

    lens_radius = aperture / 2.0;

    time0 = _time0;
    time1 = _time1;
  }

  __device__ ray get_ray(double s, double t,
                         curandState *local_rand_state) const {
    vec3 rd = lens_radius * random_in_unit_disk(local_rand_state);
    vec3 offset = u * rd.x() + v * rd.y();
    // printf("offset %f %f %f\n", offset.x(), offset.y(), offset.z());
    // direction pointing to negative direction
    return ray(origin + offset,
               lower_left_corner + s * horizontal + t * vertical - origin -
                   offset,
               random_double(time0, time1, local_rand_state));
  }

private:
  point3 origin;
  point3 lower_left_corner;
  vec3 horizontal;
  vec3 vertical;
  vec3 u, v, w;
  double lens_radius;
  double time0;
  double time1; // shutter open/close time
};
#endif