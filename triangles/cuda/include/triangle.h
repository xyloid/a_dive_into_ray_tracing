#ifndef TRIANGLE_H
#define TRIANGLE_H

#include "aabb.h"
#include "hittable.h"
#include "rtweekend.h"

class triangle : public hittable {
public:
  __device__ triangle() {}
  __device__ triangle(vec3 _v0, vec3 _v1, vec3 _v2, vec3 _vn0, vec3 _vn1,
                      vec3 _vn2, material *mat)
      : v0(_v0), v1(_v1), v2(_v2), vn0(_vn0), vn1(_vn1), vn2(_vn2),
        mat_ptr(mat) {

    // cacluate face normal
    vec3 average_vn = (vn0 + vn1 + vn2) / 3.0f;

    // counter clockwise
    AB = v1 - v0;
    AC = v2 - v0;
    vec3 face_normal_candidate = cross(AB, AC);

    face_normal = dot(face_normal_candidate, average_vn) > 0.0f
                      ? face_normal_candidate
                      : -face_normal_candidate;
    // face_normal = face_normal_candidate;

    // face normal was calculated on v0
    dist_to_origin = fabsf(dot(unit_vector(face_normal), v0));
  }

  __device__ virtual bool hit(const ray &r, float t_min, float t_max,
                              hit_record &rec,
                              curandState *local_rand_state) const override;

  __device__ virtual bool bounding_box(float time0, float time1,
                                       aabb &output_box) const override;

public:
  // A,B,C
  vec3 v0, v1, v2;
  vec3 vn0, vn1, vn2;
  // https://stackoverflow.com/questions/13689632/converting-vertex-normals-to-face-normals
  vec3 face_normal;
  float dist_to_origin;
  vec3 AB, AC;
  material *mat_ptr;
};

__device__ bool triangle::bounding_box(float time0, float time1,
                                       aabb &output_box) const {
  point3 min(fminf(fminf(v0.x(), v1.x()), v2.x()),
             fminf(fminf(v0.y(), v1.y()), v2.y()),
             fminf(fminf(v0.z(), v1.z()), v2.z()));
  point3 max(fmaxf(fmaxf(v0.x(), v1.x()), v2.x()),
             fmaxf(fmaxf(v0.y(), v1.y()), v2.y()),
             fmaxf(fmaxf(v0.z(), v1.z()), v2.z()));

  output_box = aabb(min, max);

  return true;
}

__device__ bool triangle::hit(const ray &r, float t_min, float t_max,
                              hit_record &rec,
                              curandState *local_rand_state) const {

  float norm_dot_ray_dir =
      dot(unit_vector(face_normal), unit_vector(r.direction()));

  // parallel, return false;
  if (fabsf(norm_dot_ray_dir) < 0.001) {
    return false;
  }

  // compute t
  float t = - (dot(unit_vector(face_normal), r.origin()) + dist_to_origin) /
            norm_dot_ray_dir;

  // the triangle is behind the eye
  if (t < 0) {
    return false;
  }

  rec.t = t;
  rec.p = r.origin() + rec.t * unit_vector(r.direction());

  vec3 C;

  // edge 0
  // AB = v1 - v0
  vec3 v0p = rec.p - v0;
  C = cross(AB, v0p);
  if (dot(face_normal, C) > 0) {
    // P is on the right side of AB
    return false;
  }

  // edge1 ccw
  // v2 - v1

  vec3 edge1 = v2 - v1;
  vec3 v1p = rec.p - v1;
  C = cross(edge1, v1p);
  float u = dot(face_normal, C);
  if (u > 0) {
    return false;
  }

  // edge2 ccw
  // v0 -v2
  vec3 edge2 = v0 - v2;
  vec3 v2p = rec.p - v2;
  C = cross(edge2, v2p);
  float v = dot(face_normal, C);
  if (v > 0) {
    return false;
  }

  // printf("hit %f %f %f\n", face_normal.x(), face_normal.y(), face_normal.z());

  rec.mat_ptr = mat_ptr;
  rec.u = u;
  rec.v = v;
  rec.set_face_normal(r, unit_vector(face_normal));
  return true;
}

#endif