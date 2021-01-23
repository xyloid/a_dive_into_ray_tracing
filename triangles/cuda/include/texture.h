#ifndef TEXTURE_H
#define TEXTURE_H

#include "perlin.h"
#include "rtw_stb_image.h"
#include "rtweekend.h"

class abstract_texture {
public:
  __device__ virtual color value(double u, double v, const point3 &p) const = 0;
};

class solid_color : public abstract_texture {
public:
  __device__ solid_color() {}
  __device__ solid_color(color c) : color_value(c) {}

  __device__ solid_color(double red, double green, double blue)
      : solid_color(color(red, green, blue)) {}

  __device__ virtual color value(double u, double v,
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

  __device__ virtual color value(double u, double v,
                                 const point3 &p) const override {
    double sines =
        sin(10.0 * p.x()) * sin(10.0f * p.y()) * sin(10.0 * p.z());
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

  __device__ noise_texture(double sc, curandState *local_rand_state)
      : scale(sc), noise(new perlin(local_rand_state)) {}

  __device__ virtual color value(double u, double v,
                                 const point3 &p) const override {
    // return color(1, 1, 1) * noise->noise(scale * p);
    // shift off integer values
    // return color(1, 1, 1) * 0.5 * (1.0 + noise->noise(scale * p));
    // return color(1, 1, 1) * noise->turb(scale * p);
    return color(1.0, 1.0, 1.0) * 0.5 *
           (1.0 + sin(scale * p.z() + 10.0 * noise->turb(scale * p)));
  }

public:
  perlin *noise;
  double scale;
};

class image_texture : public abstract_texture {
public:
  const static int bytes_per_pixel = 3;
  __device__ image_texture()
      : data(nullptr), width(0), height(0), bytes_per_scanline(0) {}

  __device__ image_texture(unsigned char *d, int w, int h) {
    bytes_per_scanline = bytes_per_pixel * w;
    width = w;
    height = h;
    data = d;
  }

  __device__ ~image_texture() { delete data; }

  __device__ virtual color value(double u, double v,
                                 const point3 &p) const override {
    if (data == nullptr) {
      return color(0.0, 1.0, 1.0);
    }

    u = clamp(u, 0.0, 1.0);
    v = 1.0f - clamp(v, 0.0, 1.0);

    int i = (int)(u * width);
    int j = (int)(v * height);

    if (i >= width)
      i = width - 1;
    if (j >= height)
      j = height - 1;

    // try to shift the map
    i = (i + width / 2 + width / 3) % width;

    const double color_scale = 1.0 / 255.0;
    auto pixel = data + j * bytes_per_scanline + i * bytes_per_pixel;

    return color(color_scale * pixel[0], color_scale * pixel[1],
                 color_scale * pixel[2]);
  }

public:
  unsigned char *data;
  int width, height;
  int bytes_per_scanline;
};

#endif