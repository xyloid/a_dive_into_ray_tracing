#ifndef AARECT_H
#define AARECT_H

#include "rtweekend.h"
// include to avoid "delete pointer to incomplete class"
#include "hittable.h"
#include "material.h"

// for the bounding box of the rectangle
#define THICKNESS 0.01
class xy_rect : public hittable {
public:
  __device__ xy_rect(){};

  __device__ xy_rect(float _x0, float _x1, float _y0, float _y1, float _k,
                     material *mat)
      : x0(_x0), x1(_x1), y0(_y0), y1(_y1), k(_k), mp(mat) {}

  // __device__ ~xy_rect() { delete mp; }

  __device__ virtual bool hit(const ray &r, float t_min, float t_max,
                              hit_record &rec,
                              curandState *local_rand_state) const override;

  __device__ virtual bool bounding_box(float time0, float time1,
                                       aabb &output_box) const override {

    output_box =
        aabb(point3(x0, y0, k - THICKNESS), point3(x1, y1, k + THICKNESS));
    return true;
  }

public:
  material *mp;
  float x0, x1, y0, y1, k;
};

__device__ bool xy_rect::hit(const ray &r, float t_min, float t_max,
                             hit_record &rec,
                             curandState *local_rand_state) const {
  auto t = (k - r.origin().z()) / r.direction().z();
  if (t < t_min || t > t_max) {
    return false;
  }

  auto x = r.origin().x() + t * r.direction().x();
  auto y = r.origin().y() + t * r.direction().y();

  if (x < x0 || x > x1 || y < y0 || y > y1) {
    return false;
  }

  rec.u = (x - x0) / (x1 - x0);
  rec.v = (y - y0) / (y1 - y0);
  rec.t = t;

  // default normal
  auto outward_normal = vec3(0, 0, 1);
  rec.set_face_normal(r, outward_normal);

  rec.mat_ptr = mp;
  rec.p = r.at(t);

  return true;
}

class xz_rect : public hittable {
public:
  __device__ xz_rect();

  __device__ xz_rect(float _x0, float _x1, float _z0, float _z1, float _k,
                     material *mat)
      : x0(_x0), x1(_x1), z0(_z0), z1(_z1), k(_k), mp(mat) {}

  // __device__ ~xz_rect() { delete mp; }

  __device__ virtual bool hit(const ray &r, float t_min, float t_max,
                              hit_record &rec,
                              curandState *local_rand_state) const override;

  __device__ virtual bool bounding_box(float time0, float time1,
                                       aabb &output_box) const override {
    output_box =
        aabb(point3(x0, k - THICKNESS, z0), point3(x1, k + THICKNESS, z1));
    return true;
  }

public:
  material *mp;
  float x0, x1, z0, z1, k;
};

__device__ bool xz_rect::hit(const ray &r, float t_min, float t_max,
                             hit_record &rec,
                             curandState *local_rand_state) const {
  auto t = (k - r.origin().y()) / (r.direction().y());

  if (t < t_min || t > t_max) {
    return false;
  }

  auto x = r.origin().x() + t * r.direction().x();
  auto z = r.origin().z() + t * r.direction().z();

  if (x < x0 || x > x1 || z < z0 || z > z1) {
    return false;
  }

  rec.u = (x - x0) / (x1 - x0);
  rec.v = (z - z0) / (z1 - z0);

  rec.t = t;

  auto outward_normal = vec3(0, 1, 0);

  rec.set_face_normal(r, outward_normal);
  rec.mat_ptr = mp;
  rec.p = r.at(t);

  return true;
}

class yz_rect : public hittable {
public:
  __device__ yz_rect() {}

  __device__ yz_rect(float _y0, float _y1, float _z0, float _z1, float _k,
                     material *mat)
      : y0(_y0), y1(_y1), z0(_z0), z1(_z1), k(_k), mp(mat) {}

  // __device__ ~yz_rect() { delete mp; }

  __device__ virtual bool hit(const ray &r, float t_min, float t_max,
                              hit_record &rec,
                              curandState *local_rand_state) const override;

  __device__ virtual bool bounding_box(float time0, float time1,
                                       aabb &output_box) const override {
    output_box =
        aabb(point3(k - THICKNESS, y0, z0), point3(k + THICKNESS, y1, z1));
    return true;
  }

public:
  material *mp;
  float y0, y1, z0, z1, k;
};

__device__ bool yz_rect::hit(const ray &r, float t_min, float t_max,
                             hit_record &rec,
                             curandState *local_rand_state) const {

  auto t = (k - r.origin().x()) / r.direction().x();
  if (t < t_min || t > t_max) {
    return false;
  }

  auto y = r.origin().y() + t * r.direction().y();
  auto z = r.origin().z() + t * r.direction().z();

  if (y < y0 || y > y1 || z < z0 || z > z1) {
    return false;
  }

  rec.u = (y - y0) / (y1 - y0);
  rec.v = (z - z0) / (z1 - z0);
  rec.t = t;

  auto outward_normal = vec3(1, 0, 0);

  rec.set_face_normal(r, outward_normal);
  rec.mat_ptr = mp;
  rec.p = r.at(t);

  return true;
}

#endif