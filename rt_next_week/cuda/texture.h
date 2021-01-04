#ifndef TEXTURE_H
#define TEXTURE_H

#include "perlin.h"
#include "rtweekend.h"

class abstract_texture {
public:
  __device__ virtual color value(float u, float v, const point3 &p) const = 0;
};

class solid_color : public abstract_texture {
public:
  __device__ solid_color() {}
  __device__ solid_color(color c) : color_value(c) {}

  __device__ solid_color(float red, float green, float blue)
      : solid_color(color(red, green, blue)) {}

  __device__ virtual color value(float u, float v,
                                 const point3 &p) const override {
    return color_value;
  }

public:
  color color_value;
};

class checker_texture : public abstract_texture {
public:
  __device__ checker_texture() {}

  __device__ checker_texture(abstract_texture *_even, abstract_texture *_odd)
      : even(_even), odd(_odd) {}

  __device__ checker_texture(color c1, color c2)
      : even(new solid_color(c1)), odd(new solid_color(c2)) {}

  __device__ virtual color value(float u, float v,
                                 const point3 &p) const override {
    float sines =
        sinf(10.0f * p.x()) * sinf(10.0f * p.y()) * sinf(10.0f * p.z());
    if (sines < 0)
      return odd->value(u, v, p);
    else
      return even->value(u, v, p);
  }

public:
  abstract_texture *odd;
  abstract_texture *even;
};

class noise_texture : public abstract_texture {
public:
  __device__ noise_texture() {}

  __device__ noise_texture(float sc, curandState *local_rand_state)
      : scale(sc), noise(new perlin(local_rand_state)) {}

  __device__ virtual color value(float u, float v,
                                 const point3 &p) const override {
    // return color(1, 1, 1) * noise->noise(scale * p);
    // shift off integer values
    // return color(1, 1, 1) * 0.5 * (1.0 + noise->noise(scale * p));
    // return color(1, 1, 1) * noise->turb(scale * p);
    return color(1.0f, 1.0f, 1.0f) * 0.5f *
           (1.0f + sinf(scale * p.z() + 10.0f * noise->turb(scale * p)));
  }

public:
  perlin *noise;
  float scale;
};

#endif