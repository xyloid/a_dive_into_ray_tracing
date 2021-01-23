#ifndef SPHERE_H
#define SPHERE_H

#include "aabb.h"
#include "hittable.h"

class sphere : public hittable {
public:
  __device__ sphere() {}
  __device__ sphere(vec3 cen, double r, material *m)
      : center(cen), radius(r), mat_ptr(m) {}
  // __device__ sphere(vec3 cen, double r, shared_ptr<material> m)
  //     : center(cen), radius(r), mat_ptr(m) {}

  __device__ virtual bool
  hit(const ray &r, double t_min, double t_max, hit_record &rec,
      curandState *local_rand_state) const override;
  __device__ virtual bool bounding_box(double time0, double time1,
                                       aabb &output_box) const override;

public:
  vec3 center;
  double radius;
  material *mat_ptr;
  // shared_ptr<material> mat_ptr;

protected:
  __device__ static void get_sphere_uv(const point3 &p, double &u, double &v) {
    // p: a given point on the sphere of radius one, centered at the origin.
    // u: returned value [0,1] of angle around the Y axis from X=-1.
    // v: returned value [0,1] of angle from Y=-1 to Y=+1.
    //     <1 0 0> yields <0.50 0.50>       <-1  0  0> yields <0.00 0.50>
    //     <0 1 0> yields <0.50 1.00>       < 0 -1  0> yields <0.50 0.00>
    //     <0 0 1> yields <0.25 0.50>       < 0  0 -1> yields <0.75 0.50>
    double theta = acos(-p.y());
    double phi = atan2(-p.z(), p.x()) + pi;

    u = phi / (2.0 * pi);
    v = theta / pi;
  }
};

__device__ bool sphere::hit(const ray &r, double t_min, double t_max,
                            hit_record &rec,
                            curandState *local_rand_state) const {
  vec3 oc = r.origin() - center;
  double a = dot(r.direction(), r.direction());
  double b = dot(oc, r.direction());
  double c = dot(oc, oc) - radius * radius;
  double discriminant = b * b - a * c;
  if (discriminant > 0.0) {
    double temp = (-b - sqrt(discriminant)) / a;
    if (temp < t_max && temp > t_min) {
      // find a root
      rec.t = temp;
      rec.p = r.at(rec.t);
      vec3 outward_normal = (rec.p - center) / radius;
      rec.set_face_normal(r, outward_normal);
      get_sphere_uv(outward_normal, rec.u, rec.v);
      rec.mat_ptr = mat_ptr;
      return true;
    }
    temp = (-b + sqrt(discriminant)) / a;
    if (temp < t_max && temp > t_min) {
      // find a root
      rec.t = temp;
      rec.p = r.at(rec.t);
      vec3 outward_normal = (rec.p - center) / radius;
      rec.set_face_normal(r, outward_normal);
      get_sphere_uv(outward_normal, rec.u, rec.v);
      rec.mat_ptr = mat_ptr;
      return true;
    }
  }

  return false;
}

__device__ bool sphere::bounding_box(double time0, double time1,
                                     aabb &output_box) const {
  output_box = aabb(center - vec3(radius, radius, radius),
                    center + vec3(radius, radius, radius));
  return true;
}

#endif