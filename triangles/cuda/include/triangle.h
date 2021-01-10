#ifndef TRIANGLE_H
#define TRIANGLE_H

#include "aabb.h"
#include "hittable.h"
#include "rtweekend.h"

class triangle : public hittable {
public:
  __device__ triangle() {}
  __device__ triangle(vec3 v0, vec3 v1, vec3 v2, vec3 _vn0, vec3 _vn1,
                      vec3 _vn2) {}

public:
  vec3 vertices[3];
  vec3 vertex_normal[3];
  // https://stackoverflow.com/questions/13689632/converting-vertex-normals-to-face-normals
  vec3 face_normal;
  float dist_to_origin;
};

#endif