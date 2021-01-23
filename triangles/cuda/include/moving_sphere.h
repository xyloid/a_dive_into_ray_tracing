#ifndef MOVING_SPHERE_H
#define MOVING_SPHERE_H

#include "hittable.h"
#include "rtweekend.h"
#include "sphere.h"
#include "vec3.h"

class moving_sphere : public sphere {
public:
  __device__ moving_sphere(){};
  __device__ moving_sphere(vec3 cen0, vec3 cen1, double _time0, double _time1,
                           double r, material *m)
      : sphere(cen0, r, m), center0(cen0), center1(cen1), time0(_time0),
        time1(_time1){};

  // __device__ moving_sphere(point3 cen0, point3 cen1, double _time0, double
  // _time1,
  //                          double r, shared_ptr<material> m)
  //     : sphere(cen0, r, m), center0(cen0), center1(cen1), time0(_time0),
  //       time1(_time1){};

  __device__ virtual bool hit(const ray &r, double t_min, double t_max,
                              hit_record &rec,
                              curandState *local_rand_state) const override;

  __device__ virtual bool bounding_box(double _time0, double _time1,
                                       aabb &output_box) const override;

  __device__ point3 center(double time) const;

public:
  point3 center0, center1;
  double time0, time1;
  //   double radius;
  //   material *mat_ptr;
};

__device__ point3 moving_sphere::center(double time) const {
  return center0 + ((time - time0) / (time1 - time0)) * (center1 - center0);
}

__device__ bool moving_sphere::hit(const ray &r, double t_min, double t_max,
                                   hit_record &rec,
                                   curandState *local_rand_state) const {
  vec3 oc = r.origin() - center(r.time());
  double a = r.direction().length_squared();
  double half_b = dot(oc, r.direction());
  double c = oc.length_squared() - radius * radius;

  double discriminant = half_b * half_b - a * c;
  if (discriminant < 0)
    return false;

  double sqrtd = sqrt(discriminant);

  double root = (-half_b - sqrtd) / a;
  if (root < t_min || t_max < root) {
    root = (-half_b + sqrtd) / a;
    if (root < t_min || t_max < root)
      return false;
  }

  rec.t = root;
  rec.p = r.at(rec.t);
  auto outward_normal = (rec.p - center(r.time())) / radius;
  rec.set_face_normal(r, outward_normal);
  get_sphere_uv(outward_normal, rec.u, rec.v);
  rec.mat_ptr = mat_ptr;

  return true;
}

__device__ bool moving_sphere::bounding_box(double _time0, double _time1,
                                            aabb &output_box) const {

  vec3 rad(radius, radius, radius);
  aabb box0(center(_time0) - rad, center(_time0) + rad);
  aabb box1(center(_time1) - rad, center(_time1) + rad);
  output_box = surrounding_box(box0, box1);
  return true;
}

#endif