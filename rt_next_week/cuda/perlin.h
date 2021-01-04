#ifndef PERLIN_H
#define PERLIN_H

#include "rtweekend.h"

class perlin {

public:
  __device__ perlin(curandState *local_rand_state) {
    ranfloat = new float[point_count];

    for (int i = 0; i < point_count; ++i) {
      ranfloat[i] = random_float(local_rand_state);
    }

    perm_x = perlin_generate_perm(local_rand_state);
    perm_y = perlin_generate_perm(local_rand_state);
    perm_z = perlin_generate_perm(local_rand_state);
  }

  __device__ ~perlin() {
    delete[] ranfloat;
    delete[] perm_x;
    delete[] perm_y;
    delete[] perm_z;
  }

  __device__ float noise(const point3 &p) const {
    auto i = ((int)(4.0f * p.x())) & 255;
    auto j = ((int)(4.0f * p.y())) & 255;
    auto k = ((int)(4.0f * p.z())) & 255;

    return ranfloat[perm_x[i] ^ perm_y[j] ^ perm_z[k]];
  }

private:
  static const int point_count = 256;
  float *ranfloat;
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
};

#endif
