#ifndef BVH_H
#define BVH_H

#include "hittable.h"
#include "hittable_list.h"
#include "rtweekend.h"

class bvh_node : public hittable {
public:
  __device__ bvh_node();

  __device__ bvh_node(const hittable_list &l, float time0, float time1)
      : bvh_node(l->list, 0, l->list_size, time0, time1){};

  __device__ bvh_node(const hittable **l, size_t start, size_t end, float time0,
                      float time1);

  __device__ virtual bool hit(const ray &r, float t_min, float t_max,
                              hit_record &rec) const override;

  __device__ virtual bounding_box(float time0, float time1,
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

__device__ bool bvh_node::hit(const ray &r, float r_min, float r_max,
                              hit_record &rec) const {
  if (!box.hit(r, t_min, t_max)) {
    return false;
  }
  bool hit_left = left->hit(r, t_min, t_max, rec);
  bool hit_right = right->hit(r, t_min, hit_left ? rec.t : t_max, rec);

  return hit_left || hit_right;
}



#endif