#ifndef PERLIN_H
#define PERLIN_H

#include "rtweekend.h"

class perlin {

public:
  __device__ perlin(curandState *local_rand_state) {
    // ranfloat = new float[point_count];
    ranvec = new vec3[point_count];

    for (int i = 0; i < point_count; ++i) {
      ranvec[i] = random_vec3(-1, 1, local_rand_state);
    }

    perm_x = perlin_generate_perm(local_rand_state);
    perm_y = perlin_generate_perm(local_rand_state);
    perm_z = perlin_generate_perm(local_rand_state);
  }

  __device__ ~perlin() {
    // delete[] ranfloat;
    delete[] ranvec;
    delete[] perm_x;
    delete[] perm_y;
    delete[] perm_z;
  }

  __device__ float noise(const point3 &p) const {

    // auto i = ((int)(4.0f * p.x())) & 255;
    // auto j = ((int)(4.0f * p.y())) & 255;
    // auto k = ((int)(4.0f * p.z())) & 255;

    // return ranfloat[perm_x[i] ^ perm_y[j] ^ perm_z[k]];
    float u = p.x() - floorf(p.x());
    float v = p.y() - floorf(p.y());
    float w = p.z() - floorf(p.z());

    u = u * u * (3.0f - 2.0f * u);
    v = v * v * (3.0f - 2.0f * v);
    w = w * w * (3.0f - 2.0f * w);

    int i = (int)floorf(p.x());
    int j = (int)floorf(p.y());
    int k = (int)floorf(p.z());

    // float c[2][2][2];
    vec3 c[2][2][2];

    for (int di = 0; di < 2; di++) {
      for (int dj = 0; dj < 2; dj++) {
        for (int dk = 0; dk < 2; dk++) {
          c[di][dj][dk] =
              ranvec[perm_x[(i + di) & 255] ^ perm_y[(j + dj) & 255] ^
                     perm_z[(k + dk) & 255]];
        }
      }
    }

    return trilinear_interp(c, u, v, w);
  }

private:
  static const int point_count = 256;
  // float *ranfloat;
  vec3 *ranvec;
  int *perm_x;
  int *perm_y;
  int *perm_z;

  __device__ static int *perlin_generate_perm(curandState *local_rand_state) {
    auto p = new int[point_count];

    for (int i = 0; i < point_count; i++) {
      p[i] = i;
    }

    permute(p, point_count, local_rand_state);
    return p;
  }

  __device__ __inline__ static void permute(int *p, int n,
                                            curandState *local_rand_state) {
    for (int i = n - 1; i > 0; i--) {
      // [0, n-1]
      int target = random_int(n, local_rand_state);
      int tmp = p[i];
      p[i] = p[target];
      p[target] = tmp;
    }
  }

  __device__ static float trilinear_interp(vec3 c[2][2][2], float u, float v,
                                           float w) {

    float accum = 0.0;
    for (int i = 0; i < 2; i++) {
      for (int j = 0; j < 2; j++) {
        for (int k = 0; k < 2; k++) {
          vec3 weight_v(u - i, v - j, w - k);
          accum += (i * u + (1 - i) * (1 - u)) * (j * v + (1 - j) * (1 - v)) *
                   (k * w + (1 - k) * (1 - w)) * dot(c[i][j][k], weight_v);
        }
      }
    }

    return accum;
  }
};

#endif
