#ifndef BVH_H
#define BVH_H

#include "hittable.h"
#include "hittable_list.h"
#include "rtweekend.h"
#include <algorithm>
#include <curand_kernel.h>
#include <stdio.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/sort.h>

__device__ inline bool box_compare(const hittable *a, const hittable *b,
                                   int axis) {
  aabb box_a;
  aabb box_b;
  if (!a->bounding_box(0, 0, box_a) || !b->bounding_box(0, 0, box_b)) {
    printf("No bounding box in box_compare.\n");
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

  __device__ bvh_node(hittable **l, size_t start, size_t end, float time0,
                      float time1, curandState *local_rand_state);

  __device__ virtual bool hit(const ray &r, float t_min, float t_max,
                              hit_record &rec,
                              curandState *local_rand_state) const override;

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

// __device__ bool bvh_node::hit(const ray &r, float t_min, float t_max,
//                               hit_record &rec) const {
//   if (!box.hit(r, t_min, t_max)) {
//     return false;
//   }
//   bool hit_left = left->hit(r, t_min, t_max, rec);
//   bool hit_right = right->hit(r, t_min, hit_left ? rec.t : t_max, rec);

//   return hit_left || hit_right;
// }

__device__ bool bvh_node::hit(const ray &r, float t_min, float t_max,
                              hit_record &rec,
                              curandState *local_rand_state) const {
  if (!box.hit(r, t_min, t_max)) {
    return false;
  }

  hittable *stack[256];
  hittable **stack_ptr = stack;
  *stack_ptr++ = NULL;

  bvh_node *node = (bvh_node *)this;

  bool is_hit = false;

  do {
    hittable *l_child = node->left;
    hittable *r_child = node->right;

    // TODO: working on recursion to iteration

    // no bounding box on leaf, not bvh_node
    if (l_child->is_leaf || r_child->is_leaf) {

      aabb l_box, r_box;
      l_child->bounding_box(t_min, t_max, l_box);
      r_child->bounding_box(t_min, t_max, r_box);

      // must hit one of them
      bool hit_left = l_child->hit(r, t_min, t_max, rec, local_rand_state);
      t_max = hit_left ? rec.t : t_max;
      // bool hit_right = r_child->hit(r, t_min, hit_left ? rec.t : t_max, rec,
      //                               local_rand_state);
      bool hit_right = r_child->hit(r, t_min, t_max, rec, local_rand_state);
      t_max = hit_right ? rec.t : t_max;

      node = (bvh_node *)*--stack_ptr;

      if (hit_left || hit_right)
        is_hit = true;
    } else {
      // else , we need forward to next level of the tree

      bool hit_left = ((bvh_node *)l_child)->box.hit(r, t_min, t_max);

      bool hit_right = ((bvh_node *)r_child)->box.hit(r, t_min, t_max);

      if (!hit_left && !hit_right) {
        node = (bvh_node *)*--stack_ptr;
      } else {
        node = hit_left ? (bvh_node *)l_child : (bvh_node *)r_child;
        if (hit_left && hit_right) {
          *stack_ptr++ = r_child;
        }
      }
    }

  } while (node != NULL);

  return is_hit;
}

__device__ bvh_node::bvh_node(hittable **l, size_t start, size_t end,
                              float time0, float time1,
                              curandState *local_rand_state) {
  // printf("%lu %lu enter\n", start, end);
  int axis = curand_uniform(local_rand_state) * 3;

  auto comparator =
      (axis == 0) ? box_x_compare : (axis == 1) ? box_y_compare : box_z_compare;

  size_t object_span = end - start;

  // printf("%lu %lu %d\n", start, end, axis);
  // if (object_span == 0)
  //   return;

  // printf("check\n");
  if (object_span == 1) {
    left = right = l[start];
    left->is_leaf = true;
    right->is_leaf = true;
  } else if (object_span == 2) {
    if (comparator(l[start], l[start + 1])) {
      left = l[start];
      right = l[start + 1];
    } else {
      left = l[start + 1];
      right = l[start];
    }
    left->is_leaf = true;
    right->is_leaf = true;
  } else {
    // printf("%lu %lu check\n", start, end);
    // inside kernel, using seq
    thrust::sort(thrust::seq, l + start, l + end, comparator);
    // thrust::sort(l + start, l + end, comparator);

    size_t mid = start + object_span / 2;
    // size_t mid = object_span / 2;

    // printf("%lu %lu %lu\n", start, mid, end);

    left = new bvh_node(l, start, mid, time0, time1, local_rand_state);
    // printf("%lu %lu\n", start, end);
    right = new bvh_node(l, mid, end, time0, time1, local_rand_state);
    // left = new bvh_node(l + start, 0, mid, time0, time1, local_rand_state);
    // right = new bvh_node(l + mid, 0, object_span - mid, time0, time1,
    //                      local_rand_state);
  }

  // printf("check\n");

  aabb box_left;
  aabb box_right;

  if (!left->bounding_box(time0, time1, box_left) ||
      !right->bounding_box(time0, time1, box_right)) {
    printf("No bounding box in bvh_node constructor.\n");
  }

  box = surrounding_box(box_left, box_right);
}

#endif