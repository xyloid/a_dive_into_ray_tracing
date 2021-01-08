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
  float u;
  float v;

  bool front_face;

  __device__ inline void set_face_normal(const ray &r,
                                         const vec3 &outward_normal) {
    front_face = dot(r.direction(), outward_normal) < 0;
    // NOTE : this will change the direction of the scattered rays. It has
    // effect on dielectrics. normal = front_face ? outward_normal :
    // outward_normal;
    normal = front_face ? outward_normal : -outward_normal;
  }
};

class hittable {
public:
  __device__ hittable() : is_leaf(false){};
  __device__ virtual bool hit(const ray &r, float t_min, float t_max,
                              hit_record &rec) const = 0;

  // some object (a plane) can not be bounded, so we need to return true/false
  // to indicate that.
  __device__ virtual bool bounding_box(float time0, float time1,
                                       aabb &output_box) const = 0;

public:
  bool is_leaf;
};

class translate : public hittable {
public:
  __device__ translate(hittable *p, const vec3 &displacement)
      : ptr(p), offset(displacement){};

  __device__ virtual bool hit(const ray &r, float t_min, float t_max,
                              hit_record &rec) const override;

  __device__ virtual bool bounding_box(float time0, float time1,
                                       aabb &output_box) const override;

public:
  hittable *ptr;
  vec3 offset;
};

__device__ bool translate::hit(const ray &r, float t_min, float t_max,
                               hit_record &rec) const {
  ray moved_r(r.origin() - offset, r.direction(), r.time());
  if (!ptr->hit(moved_r, t_min, t_max, rec)) {
    return false;
  }

  rec.p += offset;
  rec.set_face_normal(moved_r, rec.normal);
  return true;
}

__device__ bool translate::bounding_box(float time0, float time1,
                                        aabb &output_box) const {
  if (!ptr->bounding_box(time0, time1, output_box)) {
    return false;
  }

  output_box = aabb(output_box.min() + offset, output_box.max() + offset);

  return true;
}

#endif