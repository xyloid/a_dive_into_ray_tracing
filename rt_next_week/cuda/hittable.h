#ifndef HITTABLE_H
#define HITTABLE_H

#include "aabb.h"
#include "ray.h"


class material;

struct hit_record {
  float t;
  vec3 p;
  vec3 normal;
  material *mat_ptr;
  // shared_ptr<material> mat_ptr;
};

class hittable {
public:
  __device__ virtual bool hit(const ray &r, float t_min, float t_max,
                              hit_record &rec) const = 0;

  // some object (a plane) can not be bounded, so we need to return true/false
  // to indicate that.
  __device__ virtual bool bounding_box(float time0, float time1,
                                       aabb &output_box) const = 0;
};

#endif