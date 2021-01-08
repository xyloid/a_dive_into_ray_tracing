#ifndef CONSTANT_MEDIUM_H
#define CONSTANT_MEDIUM_H

#include "rtweekend.h"

#include "hittable.h"
#include "material.h"
#include "texture.h"

class constant_medium : public hittable {
public:
  __device__ constant_medium(hittable *b, float d, abstract_texture *a)
      : boundary(b), neg_inv_density(d), phase_function(new isotropic(a)) {}

  __device__ constant_medium(hittable *b, float d, color c)
      : boundary(b), neg_inv_density(d), phase_function(new isotropic(c)) {}

  __device__ virtual bool hit(const ray &r, float t_min, float t_max,
                              hit_record &rec,
                              curandState *local_rand_state) const override;

  __device__ virtual bool bounding_box(float time0, float time1,
                                       aabb &output_box) const override {
    return boundary->bounding_box(time0, time1, output_box);
  }

public:
  hittable *boundary;
  material *phase_function;
  float neg_inv_density;
};

__device__ bool constant_medium::hit(const ray &r, float t_min, float t_max,
                                     hit_record &rec,
                                     curandState *local_rand_state) const {
  // Print occaional samples when debugging. To enable, set enableDebug true;
  const bool enableDebug = false;

  const bool debugging =
      enableDebug && random_float(local_rand_state) < 0.00001;

  hit_record rec1, rec2;

  if (!boundary->hit(r, -infinity, infinity, rec1, local_rand_state)) {
    return false;
  }

  if (!boundary->hit(r, rec1.t + 0.0001, infinity, rec2, local_rand_state)) {
    return false;
  }

  
}

#endif