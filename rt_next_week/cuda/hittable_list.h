#ifndef HITTABLE_LIST_H
#define HITTABLE_LIST_H

#include "aabb.h"
#include "hittable.h"

class hittable_list : public hittable {
public:
  __device__ hittable_list() {}
  __device__ hittable_list(hittable **l, int n) {
    list = l;
    list_size = n;
  }

  __device__ virtual bool hit(const ray &r, float t_min, float t_max,
                              hit_record &rec) const;

  __device__ virtual bool bounding_box(float time0, float time1,
                                       aabb &output_box) const;

public:
  hittable **list;
  int list_size;
};

__device__ bool hittable_list::hit(const ray &r, float t_min, float t_max,
                                   hit_record &rec) const {
  hit_record temp_rec;
  bool hit_anything = false;
  float closest_so_far = t_max;

  for (int i = 0; i < list_size; i++) {
    if (list[i]->hit(r, t_min, closest_so_far, temp_rec)) {
      hit_anything = true;
      closest_so_far = temp_rec.t;
      rec = temp_rec;
    }
  }

  return hit_anything;
}

__device__ bool hittable_list::bounding_box(float time0, float time1,
                                            aabb &output_box) const {
  if (list_size <= 0)
    return false;
  aabb temp_box;
  bool first_box = true;

  for (int i = 0; i < list_size; i++) {
    // if next object in the list can not be bounded, return false
    if (!list[i]->bounding_box(time0, time1, temp_box))
      return false;
    // in each iteration, the box could be equal or larger. but it can not be smaller. 
    output_box = first_box ? temp_box : surrounding_box(output_box, temp_box);
    first_box = false;
  }

  return true;
}

#endif