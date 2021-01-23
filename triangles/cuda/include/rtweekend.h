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

const double infinity = std::numeric_limits<double>::infinity();
const double pi = 3.1415926535897932385f;

// Utility Functions

__device__ inline double degrees_to_radians(double degrees) {
  return degrees * pi / 180.0f;
}

// Common Headers

#include "ray.h"
#include "vec3.h"

#include <curand_kernel.h>

__device__ inline int random_int(int n, curandState *local_rand_state) {
  double val = ((double)n) - 0.000001;
  double ret = curand_uniform_double(local_rand_state) * val;
  return (int)ret;
}

__device__ inline double random_double(curandState *local_rand_state) {
  return curand_uniform_double(local_rand_state);
}

__device__ inline double random_double(double min, double max,
                                     curandState *local_rand_state) {
  return curand_uniform(local_rand_state) * (max - min) + min;
}

__device__ inline vec3 random_vec3(double min, double max,
                                   curandState *local_rand_state) {
  return vec3(random_double(min, max, local_rand_state),
              random_double(min, max, local_rand_state),
              random_double(min, max, local_rand_state));
}

__device__ inline double clamp(const double &v, const double &lo,
                              const double &hi) {
  return (v < lo) ? lo : (hi < v) ? hi : v;
}

// for the bounding box of the rectangle
#define THICKNESS 0.01

#endif