#ifndef BVH_H
#define BVH_H

#include "hittable.h"
#include "hittable_list.h"
#include "rtweekend.h"
#include <stdio.h>

__device__ inline bool box_compare(const hittable *a, const hittable *b,
                                   int axis) {
  aabb box_a;
  aabb box_b;
  if (!a->bounding_box(0, 0, box_a) || !b->bounding_box(0, 0, box_b)) {
    printf("No bounding box in bvh_node constructor.\n");
  }

  return box_a.min().e[axis] < box_b.min().e[axis];
}

__device__ bool box_x_compare(const hittable *a, const hittable *b) {
  return box_compare(a, b, 0);
}

__device__ bool box_y_compare(const hittable *a, const hittable *b) {
  return box_compare(a, b, 1);
}

__device__ bool box_z_compare(const hittable *a, const hittable *b) {
  return box_compare(a, b, 2);
}

class bvh_node : public hittable {
public:
  __device__ bvh_node();

  //   __device__ bvh_node(const hittable_list &l, float time0, float time1,
  //                       curandState *local_rand_state)
  //       : bvh_node(l.list, 0, l.list_size, time0, time1, local_rand_state){};

  __device__ bvh_node(const hittable **l, int start, int end, float time0,
                      float time1, curandState *local_rand_state);

  __device__ virtual bool hit(const ray &r, float t_min, float t_max,
                              hit_record &rec) const override;

  __device__ virtual bool bounding_box(float time0, float time1,
                                       aabb &output_box) const override;

public:
  hittable *left;
  hittable *right;
  aabb box;
};

__device__ bool bvh_node::bounding_box(float time0, float time1,
                                       aabb &output_box) const {
  output_box = box;
  return true;
}

__device__ bool bvh_node::hit(const ray &r, float t_min, float t_max,
                              hit_record &rec) const {
  if (!box.hit(r, t_min, t_max)) {
    return false;
  }
  bool hit_left = left->hit(r, t_min, t_max, rec);
  bool hit_right = right->hit(r, t_min, hit_left ? rec.t : t_max, rec);

  return hit_left || hit_right;
}

__device__ bvh_node::bvh_node(const hittable **l, int start, int end, float time0,
                    float time1, curandState *local_rand_state) {
  int axis = curand_uniform(local_rand_state) * 3;

  auto comparator =
      (axis == 0) ? box_x_compare : (axis == 1) ? box_y_compare : box_z_compare;

  size_t object_span = end - start;
}

#endif