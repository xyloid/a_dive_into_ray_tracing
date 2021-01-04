#ifndef RTWEEKEND_H
#define RTWEEKEND_H

#include <cmath>
#include <limits>
#include <memory>

// Usings

using std::make_shared;
using std::shared_ptr;
using std::sqrt;

// Constants

const float infinity = std::numeric_limits<float>::infinity();
const float pi = 3.1415926535897932385f;

// Utility Functions

__device__ inline float degrees_to_radians(float degrees) {
  return degrees * pi / 180.0f;
}

// Common Headers

#include "ray.h"
#include "vec3.h"

#include <curand_kernel.h>

__device__ inline int random_int(int n, curandState *local_rand_state) {
  float val = ((float)n) - 0.000001;
  float ret = curand_uniform(local_rand_state) * val;
  return (int)ret;
}

__device__ inline float random_float(curandState *local_rand_state) {
  return curand_uniform(local_rand_state);
}

__device__ inline float random_float(float min, float max,
                                     curandState *local_rand_state) {
  return curand_uniform(local_rand_state) * (max - min) + min;
}

__device__ inline vec3 random_vec3(float min, float max,
                                   curandState *local_rand_state) {
  return vec3(random_float(min, max, local_rand_state),
              random_float(min, max, local_rand_state),
              random_float(min, max, local_rand_state));
}

__device__ inline float clamp(const float &v, const float &lo,
                              const float &hi) {
  return (v < lo) ? lo : (hi < v) ? hi : v;
}

#endif