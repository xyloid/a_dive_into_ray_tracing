#ifndef HITTABLE_H
#define HITTABLE_H

#include "aabb.h"
#include "ray.h"
#include "rtweekend.h"
#include <curand_kernel.h>

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
    // normal = front_face ? outward_normal : -outward_normal;
    normal = front_face ? outward_normal : outward_normal;
  }
};

class hittable {
public:
  __device__ hittable() : is_leaf(false){};
  __device__ virtual bool hit(const ray &r, float t_min, float t_max,
                              hit_record &rec,
                              curandState *local_rand_state) const = 0;

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

  __device__ virtual bool
  hit(const ray &r, float t_min, float t_max, hit_record &rec,
      curandState *local_rand_state = NULL) const override;

  __device__ virtual bool bounding_box(float time0, float time1,
                                       aabb &output_box) const override;

public:
  hittable *ptr;
  vec3 offset;
};

__device__ bool translate::hit(const ray &r, float t_min, float t_max,
                               hit_record &rec,
                               curandState *local_rand_state) const {
  // offset the ray, compare this with the code that offsets the bounding box,
  // they have different reference coordinate frames.
  ray moved_r(r.origin() - offset, r.direction(), r.time());
  if (!ptr->hit(moved_r, t_min, t_max, rec, local_rand_state)) {
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

  // offset the bounding_box
  output_box = aabb(output_box.min() + offset, output_box.max() + offset);

  return true;
}

class rotate_y : public hittable {
public:
  __device__ rotate_y(hittable *p, float angle);

  __device__ virtual bool hit(const ray &r, float t_min, float t_max,
                              hit_record &rec,
                              curandState *local_rand_state) const override;

  __device__ virtual bool bounding_box(float time0, float time1,
                                       aabb &output_box) const override {
    output_box = bbox;
    return hasbox;
  }

public:
  hittable *ptr;
  float sin_theta;
  float cos_theta;
  bool hasbox;
  aabb bbox;
};

__device__ rotate_y::rotate_y(hittable *p, float angle) : ptr(p) {
  float radians = degrees_to_radians(angle);

  sin_theta = sinf(radians);
  cos_theta = cosf(radians);

  // note time0 and time1 values may need to change
  hasbox = ptr->bounding_box(0, 1, bbox);

  // offset the bounding_box;
  point3 min(infinity, infinity, infinity);
  point3 max(-infinity, -infinity, -infinity);

  for (int i = 0; i < 2; i++) {
    for (int j = 0; j < 2; j++) {
      for (int k = 0; k < 2; k++) {

        // i,j,k in {0,1}
        // this loop generate 8 vertices' coordinates
        // and then rotate the vector and find the max and min values

        float x = i * bbox.max().x() + (1 - i) * bbox.min().x();
        float y = j * bbox.max().y() + (1 - j) * bbox.min().y();
        float z = k * bbox.max().z() + (1 - k) * bbox.min().z();

        float newx = cos_theta * x + sin_theta * z;
        float newz = -sin_theta * x + cos_theta * z;

        vec3 tester(newx, y, newz);

        for (int c = 0; c < 3; c++) {
          min[c] = fminf(min[c], tester[c]);
          max[c] = fmaxf(max[c], tester[c]);
        }
      }
    }
  }

  bbox = aabb(min, max);
}

__device__ bool rotate_y::hit(const ray &r, float t_min, float t_max,
                              hit_record &rec,
                              curandState *local_rand_state) const {
  // offset the ray

  auto origin = r.origin();
  auto direction = r.direction();

  // think of it as offset -angle
  origin[0] = cos_theta * r.origin()[0] - sin_theta * r.origin()[2];
  origin[2] = sin_theta * r.origin()[0] + cos_theta * r.origin()[2];

  direction[0] = cos_theta * r.direction()[0] - sin_theta * r.direction()[2];
  direction[2] = sin_theta * r.direction()[0] + cos_theta * r.direction()[2];

  ray rotated_r(origin, direction, r.time());

  if (!ptr->hit(rotated_r, t_min, t_max, rec, local_rand_state)) {
    return false;
  }

  auto p = rec.p;
  auto normal = rec.normal;

  p[0] = cos_theta * rec.p[0] + sin_theta * rec.p[2];
  p[2] = -sin_theta * rec.p[0] + cos_theta * rec.p[2];

  normal[0] = cos_theta * rec.normal[0] + sin_theta * rec.normal[2];
  normal[2] = -sin_theta * rec.normal[0] + cos_theta * rec.normal[2];

  rec.p = p;
  rec.set_face_normal(rotated_r, normal);

  return true;
}

#endif