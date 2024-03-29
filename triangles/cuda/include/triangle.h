#ifndef TRIANGLE_H
#define TRIANGLE_H

#include "aabb.h"
#include "hittable.h"
#include "rtweekend.h"
#include <float.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

class triangle : public hittable {
public:
  __device__ __host__ triangle() {}
  __device__ __host__ triangle(vec3 _v0, vec3 _v1, vec3 _v2, vec3 _vn0,
                               vec3 _vn1, vec3 _vn2, material *mat)
      : v0(_v0), v1(_v1), v2(_v2), vn0(_vn0), vn1(_vn1), vn2(_vn2),
        mat_ptr(mat) {

    // mat_ptr = mat;
    // cacluate face normal
    vec3 average_vn = -(vn0 + vn1 + vn2) / 3.0f;

    // counter clockwise
    AB = v1 - v0;
    AC = v2 - v0;

    vec3 face_normal_candidate = cross(AB, AC);

    // if (dot(face_normal_candidate, average_vn) < 0.0f) {
    //   v0 = _v2;
    //   v2 = _v0;
    //   vn0 = _vn2;
    //   vn2 = _vn0;
    // }

    // AB = v1 - v0;
    // AC = v2 - v0;

    face_normal = dot(face_normal_candidate, average_vn) > 0.0f
                      ? face_normal_candidate
                      : -face_normal_candidate;
    // face_normal = (vn0 + vn1 + vn2) / 3.0f;

    // face_normal = cross(AB, AC);

    face_normal_unit = unit_vector(face_normal);

    // face normal was calculated on v0
    dist_to_origin = fabsf(dot(face_normal_unit, v0));
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
  vec3 face_normal_unit;
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

  // if (fabsf(min.x() - max.x()) < THICKNESS) {
  //   min.e[0] = min.x() - THICKNESS;
  //   max.e[0] = max.x() + THICKNESS;
  // }

  for (int i = 0; i < 3; i++) {
    if (fabsf(min.e[i] - max.e[i]) < THICKNESS) {
      min.e[i] = min.e[i] - THICKNESS;
      max.e[i] = max.e[i] + THICKNESS;
    }
    // min.e[i] = min.e[i] - THICKNESS;
    // max.e[i] = max.e[i] + THICKNESS;
  }

  output_box = aabb(min, max);

  return true;
}

__device__ bool triangle::hit(const ray &r, float t_min, float t_max,
                              hit_record &rec,
                              curandState *local_rand_state) const {

  // printf("enter %f, %f, %f,  -  %f, %f, %f\n", r.orig.x(), r.orig.y(),
  //        r.orig.z(), r.dir.x(), r.dir.y(), r.dir.z());

  // check if the light shoot from out side, I want to see if there is a lot
  // bouncing in side the box.

  // comment out and missing triangle reduced
  // if (dot(r.dir, face_normal_unit) < 0.0) {
  //   // value > 0, then the face normal and ray has same direction, should not
  //   // bouncing then.
  //   return false;
  // }

  vec3 ray_dir_unit = unit_vector(r.direction());

  // float norm_dot_ray_dir = dot(face_normal_unit, ray_dir_unit);

  float norm_dot_ray_dir = dot(face_normal, r.direction());

  // parallel, return false;
  if (fabsf(norm_dot_ray_dir) < 0.01) {
    return false;
  }

  // compute t
  // float t =
  //     -(dot(face_normal_unit, r.origin()) + dist_to_origin) /
  //     norm_dot_ray_dir;

  float t = dot(v0 - r.origin(), face_normal) / norm_dot_ray_dir;

  // the triangle is behind the eye
  if (t < 0) {
    // printf("t < 0\n");
    return false;
  }

  // we can not use t_min and t_max here since the t has different unit.
  // if (t < t_min || t > t_max) {
  //   return false;
  // }

  // vec3 p = r.origin() + t * ray_dir_unit;
  vec3 p = r.at(t);

  // vec3 dist = p - r.origin();

  // // convert t to the comparable value to other shape primitives
  // // t should be used determine which p is at the front.
  // float converted_t = dist.length() / r.direction().length();

  // if (converted_t < t_min || converted_t > t_max) {
  //   return false;
  // }

    if (t < t_min || t > t_max) {
    return false;
  }

  vec3 C;

  // strange missing triangles
  const float threshold = 0;

  // edge 0
  // AB = v1 - v0
  vec3 v0p = p - v0;
  C = cross(v1 - v0, v0p);
  if (dot(face_normal_unit, C) < threshold) {
    // P is on the right side of AB
    // printf("return 1\n");

    return false;
  }

  // edge1 ccw
  // v2 - v1

  vec3 edge1 = v2 - v1;
  vec3 v1p = p - v1;
  C = cross(edge1, v1p);
  float u = dot(face_normal_unit, C);
  if (u < threshold) {
    // printf("return 2\n");
    return false;
  }

  // edge2 ccw
  // v0 -v2
  vec3 edge2 = v0 - v2;
  vec3 v2p = p - v2;
  C = cross(edge2, v2p);
  float v = dot(face_normal_unit, C);
  if (v < threshold) {
    // printf("return 3\n");
    return false;
  }

  // printf("hit %f %f %f\n", rec.p.x(), rec.p.y(), rec.p.z());
  // this t is used in bvh traversal
  // rec.t = converted_t;
   rec.t = t;
  rec.p = p;
  rec.u = u;
  rec.v = v;
  rec.set_face_normal(r, face_normal_unit);
  // rec.set_face_normal(r, face_normal);
  rec.mat_ptr = mat_ptr;
  return true;
}

void read_triangles(std::vector<triangle> &triangles) {
  std::vector<vec3> vns;
  std::vector<vec3> vs;
  // std::string filename = "objs/dafault_cube_in_triangles.obj";
  // std::string filename = "objs/bunny.obj";
  // std::string filename = "objs/ball_in_triangles.obj";
  // std::string filename = "objs/bunny_s_blender_10.obj";
  // std::string filename = "objs/bunny_blender.obj";
  // std::string filename = "objs/bunny_s_blender_20.obj";
  // std::string filename = "objs/bunny_s_blender_20000.obj";
  // std::string filename = "objs/bunny_s_blender_1000.obj";
  // std::string filename = "objs/bunny_s.obj";
  // std::ifstream infile("objs/test.obj");
  std::string filename = "objs/blender_monkey.obj";

  // std::string filename = "objs/bunny_large.obj";

  std::ifstream infile(filename);

  if (infile.is_open()) {
    std::string line;
    double x, y, z;
    while (std::getline(infile, line)) {

      std::istringstream in(line);
      std::string type;
      in >> type;
      if (type == "vn") {
        in >> x >> y >> z;
        vns.push_back(vec3(x, y, z));
        // std::cout << "vn found " << std::endl
        //           << line << std::endl
        //           << x << "," << y << "," << z << std::endl;
      } else if (type == "v") {
        in >> x >> y >> z;
        vs.push_back(vec3(x, y, z));
        // std::cout << "v found " << std::endl
        //           << line << std::endl
        //           << x << "," << y << "," << z << std::endl;
      } else if (type == "f") {
        // find face
        // 0/1/2 3/4/5 6/7/8
        //
        // format 1//1 2//2 3//2
        std::vector<int> indices;
        while (!in.eof()) {
          std::string section;
          in >> section;
          // std::cout << "section: " << section << std::endl;
          char delimiter = '/';
          std::istringstream sec(section);
          std::string num;
          while (std::getline(sec, num, delimiter)) {
            if (num.length() == 0) {
              indices.push_back(-1);
            } else {
              // this is an integer
              int n = std::stoi(num) - 1;
              // std::cout << num << "\t" << n << std::endl;
              indices.push_back(n);
            }
          }
        }
        // cw to ccw
        // triangles.push_back(triangle(vs.at(indices.at(6)),
        // vs.at(indices.at(3)),
        //                              vs.at(indices.at(0)),
        //                              vs.at(indices.at(5)),
        //                              vs.at(indices.at(2)),
        //                              vs.at(indices.at(8)), nullptr));

        // std::cout<<vs.at(indices.at(0))<<std::endl;
        triangles.push_back(
            triangle(vs.at(indices.at(6)), vs.at(indices.at(3)),
                     vs.at(indices.at(0)), vns.at(indices.at(8)),
                     vns.at(indices.at(5)), vns.at(indices.at(2)), nullptr));
      }
    }
    infile.close();
  } else {

    std::cerr << "read failed" << std::endl;
  }
}

#endif