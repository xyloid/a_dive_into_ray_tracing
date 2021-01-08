#ifndef BOX_H
#define BOX_H

#include "rtweekend.h"

#include "aarect.h"
#include "hittable.h"
#include "hittable_list.h"

class box : public hittable {
public:
  __device__ box() {}

  __device__ box(const point3 &p0, const point3 &p1, material *ptr);

  __device__ ~box() {
    for (int i = 0; i < sides->list_size; i++) {
      delete sides->list[i];
    }
    delete[] sides->list;
    delete sides;
  }

  __device__ virtual bool hit(const ray &r, float t_min, float t_max,
                              hit_record &rec,
                              curandState *local_rand_state) const override;

  __device__ virtual bool bounding_box(float time0, float time1,
                                       aabb &output_box) const override {
    // diagnal of the box
    output_box = aabb(box_min, box_max);
    return true;
  };

public:
  point3 box_min;
  point3 box_max;
  hittable_list *sides;
};

__device__ box::box(const point3 &p0, const point3 &p1, material *ptr) {
  box_min = p0;
  box_max = p1;

  // create an array on the heap
  hittable **objects = new hittable *[6];

  objects[0] = new xy_rect(p0.x(), p1.x(), p0.y(), p1.y(), p1.z(), ptr);
  objects[1] = new xy_rect(p0.x(), p1.x(), p0.y(), p1.y(), p0.z(), ptr);

  objects[2] = new xz_rect(p0.x(), p1.x(), p0.z(), p1.z(), p1.y(), ptr);
  objects[3] = new xz_rect(p0.x(), p1.x(), p0.z(), p1.z(), p0.y(), ptr);

  objects[4] = new yz_rect(p0.y(), p1.y(), p0.z(), p1.z(), p1.x(), ptr);
  objects[5] = new yz_rect(p0.y(), p1.y(), p0.z(), p1.z(), p0.x(), ptr);

  sides = new hittable_list(objects, 6);
}

__device__ bool box::hit(const ray &r, float t_min, float t_max,
                         hit_record &rec, curandState *local_rand_state) const {
  return sides->hit(r, t_min, t_max, rec, local_rand_state);
}

#endif