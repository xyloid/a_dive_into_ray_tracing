#ifndef SPHERE_H
#define SPHERE_H

#include "aabb.h"
#include "hittable.h"

class sphere : public hittable {
public:
  __device__ sphere() {}
  __device__ sphere(vec3 cen, float r, material *m)
      : center(cen), radius(r), mat_ptr(m) {}
  // __device__ sphere(vec3 cen, double r, shared_ptr<material> m)
  //     : center(cen), radius(r), mat_ptr(m) {}

  __device__ virtual bool hit(const ray &r, float t_min, float t_max,
                              hit_record &rec) const override;
  __device__ virtual bool bounding_box(float time0, float time1,
                                       aabb &output_box) const override;

public:
  vec3 center;
  float radius;
  material *mat_ptr;
  // shared_ptr<material> mat_ptr;

protected:
  __device__ static void get_sphere_uv(const point3 &p, float &u, float &v) {
    // p: a given point on the sphere of radius one, centered at the origin.
    // u: returned value [0,1] of angle around the Y axis from X=-1.
    // v: returned value [0,1] of angle from Y=-1 to Y=+1.
    //     <1 0 0> yields <0.50 0.50>       <-1  0  0> yields <0.00 0.50>
    //     <0 1 0> yields <0.50 1.00>       < 0 -1  0> yields <0.50 0.00>
    //     <0 0 1> yields <0.25 0.50>       < 0  0 -1> yields <0.75 0.50>
    float theta = acos(-p.y());
    float phi = atan2(-p.z(), p.x()) + pi;

    u = phi / (2.0f * pi);
    v = theta / pi;
  }
};

__device__ bool sphere::hit(const ray &r, float t_min, float t_max,
                            hit_record &rec) const {
  vec3 oc = r.origin() - center;
  float a = dot(r.direction(), r.direction());
  float b = dot(oc, r.direction());
  float c = dot(oc, oc) - radius * radius;
  float discriminant = b * b - a * c;
  if (discriminant > 0.0f) {
    float temp = (-b - sqrt(discriminant)) / a;
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

__device__ bool sphere::bounding_box(float time0, float time1,
                                     aabb &output_box) const {
  output_box = aabb(center - vec3(radius, radius, radius),
                    center + vec3(radius, radius, radius));
  return true;
}

#endif