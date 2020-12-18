#ifndef SPHERE_H
#define SPHERE_H

#include "hittable.h"

class sphere : public hittable {
public:
  __device__ sphere() {}
  __device__ sphere(vec3 cen, float r) : center(cen), radius(r) {}

  __device__ virtual bool hit(const ray &r, float t_min, float t_max,
                              hit_record &rec) const;

public:
  vec3 center;
  float radius;
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
      rec.normal = (rec.p - center) / radius;
      return true;
    }
    temp = (-b + sqrt(discriminant)) / a;
    if (temp < t_max && temp > t_min) {
      // find a root
      rec.t = temp;
      rec.p = r.at(rec.t);
      rec.normal = (rec.p - center) / radius;
      return true;
    }
  }

  return false;
}

#endif