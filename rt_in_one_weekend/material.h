#ifndef MATERIAL_H
#define MATERIAL_H

#include "hittable.h"
#include "rtweekend.h"

struct hit_record;

class material {
public:
  virtual bool scatter(const ray &r_in, const hit_record &rec,
                       color &attenuation, ray &scattered) const = 0;
};

class lambertian : public material {
public:
  lambertian(const color &a) : albedo(a){};

  virtual bool scatter(const ray &r_in, const hit_record &rec,
                       color &attenuation, ray &scattered) const override {
    // scatter_direction could be 0
    auto scatter_direction = rec.normal + random_unit_vector();

    if (scatter_direction.near_zero()) {
      scatter_direction = rec.normal;
    }

    scattered = ray(rec.p, scatter_direction);
    attenuation = albedo;
    return true;
  };

public:
  color albedo;
};

class metal : public material {
public:
  metal(const color &a, double f) : albedo(a), fuzz(f < 1 ? f : 1) {}
  virtual bool scatter(const ray &r_in, const hit_record &rec,
                       color &attenuation, ray &scattered) const override {
    vec3 reflected = reflect(unit_vector(r_in.direction()), rec.normal);
    // new ray. start from intersection point, pointing to refelcted direction
    // scattered = ray(rec.p, reflected);
    // add fuzz
    scattered = ray(rec.p, reflected + fuzz * random_in_unit_sphere());
    attenuation = albedo;
    return (dot(scattered.direction(), rec.normal) > 0);
  }

public:
  color albedo;
  double fuzz;
};

class dielectric : public material {
public:
  dielectric(double index_of_refraction) : ir(index_of_refraction) {}

  virtual bool scatter(const ray &r_in, const hit_record &rec,
                       color &attenuation, ray &scattered) const override {
    attenuation = color(1.0, 1.0, 1.0);
    // assuming one side is always air, air ir = 1.0
    double refraction_ratio = rec.front_face ? (1.0 / ir) : ir;

    vec3 unit_direction = unit_vector(r_in.direction());
    vec3 refracted = refract(unit_direction, rec.normal, refraction_ratio);
    scattered = ray(rec.p, refracted);
    return true;
  }

public:
  double ir; // Index of Refaction
};

#endif