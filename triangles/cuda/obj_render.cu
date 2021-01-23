#include "aarect.h"
#include "box.h"
#include "bvh.h"
#include "camera.h"
#include "constant_medium.h"
#include "cuda_utils.h"
#include "hittable_list.h"
#include "material.h"
#include "moving_sphere.h"
#include "ray.h"
#include "sphere.h"
#include "triangle.h"
#include "vec3.h"
#include <cuda.h>
#include <curand_kernel.h>
#include <float.h>
#include <iostream>
#include <time.h>

__device__ vec3 get_color(const ray &r, color **background, hittable **world,
                          curandState *local_rand_state) {
  ray cur_ray = r;
  vec3 cur_attenuation(1.0f, 1.0f, 1.0f);

  const int depth = 50;

  vec3 emitted_rec[depth];
  vec3 attenuation_rec[depth];
  int i = 0;
  for (i = 0; i < depth; i++) {
    hit_record rec;
    if ((*world)->hit(cur_ray, 0.00001f, FLT_MAX, rec, local_rand_state)) {

      ray scattered;
      vec3 attenuation;

      color emitted = rec.mat_ptr->emitted(rec.u, rec.v, rec.p);

      if (rec.mat_ptr->scatter(cur_ray, rec, attenuation, scattered,
                               local_rand_state)) {
        // scattered
        // cur_attenuation *= attenuation;
        // cur_attenuation += emitted;
        // cur_attenuation *= (attenuation + emitted);
        emitted_rec[i] = emitted;
        attenuation_rec[i] = attenuation;

        cur_ray = scattered;

      } else {
        // no scatter
        // no attenuation
        // no background light
        // but we have emitted

        cur_attenuation *= emitted;

        while (i-- > 0) {
          cur_attenuation =
              emitted_rec[i] + cur_attenuation * attenuation_rec[i];
        }

        return cur_attenuation;
      }
    } else {
      cur_attenuation *= **background;
      while (i-- > 0) {
        cur_attenuation = emitted_rec[i] + cur_attenuation * attenuation_rec[i];
      }
      return cur_attenuation;
    }
  }

  // no hit
  // only have background
  // cur_attenuation *= vec3(0, 0, 0.73);
  cur_attenuation *= **background;
  while (i-- > 0) {
    cur_attenuation = emitted_rec[i] + cur_attenuation * attenuation_rec[i];
  }

  return cur_attenuation;
  // return **background; // exceeded recursion
  // return vec3(0, 0, 0.73); // exceeded recursion
}

__global__ void rand_init(curandState *rand_state) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    curand_init(1984, 3, 17, rand_state);
  }
}

__global__ void render_init(int max_x, int max_y, curandState *rand_state) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;
  if ((i >= max_x) || (j >= max_y))
    return;
  int pixel_index = j * max_x + i;
  // Original: Each thread gets same seed, a different sequence number, no
  // offset curand_init(1984, pixel_index, 0, &rand_state[pixel_index]); BUGFIX,
  // see Issue#2: Each thread gets different seed, same sequence for performance
  // improvement of about 2x!
  // curand_init(1984 + pixel_index, 0, 0, &rand_state[pixel_index]);
  curand_init(1984 + pixel_index, i, j, &rand_state[pixel_index]);
}

__global__ void render(vec3 *fb, int max_x, int max_y, int ns, camera **cam,
                       hittable **world, curandState *rand_state,
                       color **background) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;
  if ((i >= max_x) || (j >= max_y))
    return;
  int pixel_index = j * max_x + i;
  curandState *local_rand_state = &rand_state[pixel_index];
  vec3 col(0, 0, 0);
  for (int s = 0; s < ns; s++) {
    float u = float(i + curand_uniform(local_rand_state)) / float(max_x);
    float v = float(j + curand_uniform(local_rand_state)) / float(max_y);
    ray r = (*cam)->get_ray(u, v, local_rand_state);
    col += get_color(r, background, world, local_rand_state);
  }
  rand_state[pixel_index] = *local_rand_state;
  col /= float(ns);
  col[0] = sqrt(col[0]);
  col[1] = sqrt(col[1]);
  col[2] = sqrt(col[2]);
  fb[pixel_index] = col;
}

#define RND (curand_uniform(local_rand_state))

__device__ hittable *random_scene(hittable **d_list,
                                  curandState *local_rand_state) {
  auto checker =
      new checker_texture(color(0.2, 0.3, 0.1), color(0.9, 0.9, 0.9));

  d_list[0] = new sphere(vec3(0, -1000.0, -1), 1000, new lambertian(checker));
  // d_list[0] = new sphere(vec3(0, -1000.0, -1), 1000,
  //                        new lambertian(vec3(0.5, 0.5, 0.5)));
  // d_list[0] = new sphere(vec3(0, -1000.0, -1), 1000,
  //                    make_shared<lambertian>(vec3(0.5, 0.5, 0.5)));
  int i = 1;
  for (int a = -11; a < 11; a++) {
    for (int b = -11; b < 11; b++) {
      float choose_mat = RND;
      vec3 center(a + RND, 0.2, b + RND);
      if (choose_mat < 0.8f) {

        vec3 center2 = center + vec3(0, RND * 0.5f, 0);
        // d_list[i++] =
        //     new sphere(center, 0.2,
        //                new lambertian(vec3(RND * RND, RND * RND, RND *
        //                RND)));
        d_list[i++] = new moving_sphere(
            center, center2, 0.0, 1.0, 0.2,
            new lambertian(vec3(RND * RND, RND * RND, RND * RND)));

      } else if (choose_mat < 0.95f) {
        d_list[i++] =
            new sphere(center, 0.2,
                       new metal(vec3(0.5f * (1.0f + RND), 0.5f * (1.0f + RND),
                                      0.5f * (1.0f + RND)),
                                 0.5f * RND));
      } else {
        d_list[i++] = new sphere(center, 0.2, new dielectric(1.5));
      }
    }
  }
  d_list[i++] = new sphere(vec3(0, 1, 0), 1.0, new dielectric(1.5));
  d_list[i++] =
      new sphere(vec3(-4, 1, 0), 1.0, new lambertian(vec3(0.4, 0.2, 0.1)));
  d_list[i++] =
      new sphere(vec3(4, 1, 0), 1.0, new metal(vec3(0.7, 0.6, 0.5), 0.0));

  return new bvh_node(d_list, 0, 22 * 22 + 1 + 3, 0.0f, 1.0f, local_rand_state);
}

__device__ hittable *two_spheres(curandState *local_rand_state) {

  auto checker =
      new checker_texture(color(0.2, 0.3, 0.1), color(0.9, 0.9, 0.9));

  hittable *ret[2];
  ret[0] = new sphere(point3(0, -10, 0), 10, new lambertian(checker));
  ret[1] = new sphere(point3(0, 10, 0), 10, new lambertian(checker));

  return new bvh_node(ret, 0, 2, 0.0f, 1.0f, local_rand_state);
}

__device__ hittable *two_perlin_spheres(curandState *local_rand_state) {

  auto perlin_texture = new noise_texture(4, local_rand_state);

  hittable *ret[2];
  ret[0] =
      new sphere(point3(0, -1000, 0), 1000, new lambertian(perlin_texture));
  ret[1] = new sphere(point3(0, 2, 0), 2, new lambertian(perlin_texture));

  return new bvh_node(ret, 0, 2, 0.0f, 1.0f, local_rand_state);
}

__device__ hittable *earth(unsigned char *data, int w, int h,
                           curandState *local_rand_state) {
  auto earth_texture = new image_texture(data, w, h);
  auto earth_surface = new lambertian(earth_texture);

  hittable *ret[1];
  ret[0] = new sphere(point3(0, 0, 0), 2, earth_surface);
  return new bvh_node(ret, 0, 1, 0.0f, 1.0f, local_rand_state);
}

__device__ hittable *simple_light(curandState *local_rand_state) {
  auto perlin_texture = new noise_texture(4, local_rand_state);

  hittable *ret[4];
  ret[0] =
      new sphere(point3(0, -1000, 0), 1000, new lambertian(perlin_texture));
  ret[1] = new sphere(point3(0, 2, 0), 2, new lambertian(perlin_texture));

  auto diff_light = new diffuse_light(color(4, 4, 4));

  ret[2] = new xy_rect(3, 5, 1, 2, -2, diff_light);

  auto diff_light2 = new diffuse_light(color(6, 4, 4));
  ret[3] = new sphere(point3(0, 6, 0), 1.5, diff_light2);

  return new bvh_node(ret, 0, 4, 0.0f, 1.0f, local_rand_state);
}

__device__ hittable *cornell_box(curandState *local_rand_state) {
  hittable *ret[8];
  auto red = new lambertian(color(.65, .05, .05));
  auto white = new lambertian(color(.73, .73, .73));
  auto green = new lambertian(color(.12, .45, .15));
  auto light = new diffuse_light(color(15, 15, 15));

  ret[0] = new yz_rect(0, 555, 0, 555, 555, green);
  ret[1] = new yz_rect(0, 555, 0, 555, 0, red);
  ret[2] = new xz_rect(213, 343, 227, 332, 554, light);
  ret[3] = new xz_rect(0, 555, 0, 555, 0, white);
  ret[4] = new xz_rect(0, 555, 0, 555, 555, white);
  ret[5] = new xy_rect(0, 555, 0, 555, 555, white);

  // ret[6] = new box(point3(130, 0, 65), point3(295, 165, 230), white);
  // ret[7] = new box(point3(265, 0, 295), point3(430, 330, 460), white);

  hittable *box1 = new box(point3(0, 0, 0), point3(165, 330, 165), white);
  box1 = new rotate_y(box1, 15);
  box1 = new translate(box1, vec3(265, 0, 295));

  hittable *box2 = new box(point3(0, 0, 0), point3(165, 165, 165), white);
  box2 = new rotate_y(box2, -18);
  box2 = new translate(box2, vec3(130, 0, 65));

  ret[6] = box1;
  ret[7] = box2;

  return new bvh_node(ret, 0, 8, 0.0f, 1.0f, local_rand_state);
}

__device__ hittable *cornell_smoke(curandState *local_rand_state) {
  hittable *ret[8];
  auto red = new lambertian(color(.65, .05, .05));
  auto white = new lambertian(color(.73, .73, .73));
  auto green = new lambertian(color(.12, .45, .15));
  auto light = new diffuse_light(color(15, 15, 15));

  ret[0] = new yz_rect(0, 555, 0, 555, 555, green);
  ret[1] = new yz_rect(0, 555, 0, 555, 0, red);
  ret[2] = new xz_rect(213, 343, 227, 332, 554, light);
  ret[3] = new xz_rect(0, 555, 0, 555, 0, white);
  ret[4] = new xz_rect(0, 555, 0, 555, 555, white);
  ret[5] = new xy_rect(0, 555, 0, 555, 555, white);

  hittable *box1 = new box(point3(0, 0, 0), point3(165, 330, 165), white);
  box1 = new rotate_y(box1, 15);
  box1 = new translate(box1, vec3(265, 0, 295));
  box1 = new constant_medium(box1, 0.01, color(0, 0, 0));

  hittable *box2 = new box(point3(0, 0, 0), point3(165, 165, 165), white);
  box2 = new rotate_y(box2, -18);
  box2 = new translate(box2, vec3(130, 0, 65));
  box2 = new constant_medium(box2, 0.01, color(1, 1, 1));

  ret[6] = box1;
  ret[7] = box2;

  return new bvh_node(ret, 0, 8, 0.0f, 1.0f, local_rand_state);
}

__device__ hittable *rt_next_week_final_scene(unsigned char *data, int w, int h,
                                              curandState *local_rand_state) {
  const int boxes_per_side = 20;

  const int num_obj = boxes_per_side * boxes_per_side + 10;
  hittable *ret[num_obj];

  // ground
  auto ground = new lambertian(color(0.48, 0.83, 0.53));

  int index = 0;
  for (int i = 0; i < boxes_per_side; i++) {
    for (int j = 0; j < boxes_per_side; j++) {
      float w = 100.0;
      float x0 = -1000.0f + i * w;
      float z0 = -1000.0f + j * w;
      float y0 = 0.0;
      float x1 = x0 + w;
      float y1 = random_float(1, 101, local_rand_state);
      float z1 = z0 + w;
      ret[index++] = new box(point3(x0, y0, z0), point3(x1, y1, z1), ground);
    }
  }

  // light
  auto light = new diffuse_light(color(7, 7, 7));
  ret[index++] = new xz_rect(123, 423, 147, 412, 554, light);

  // moving sphere
  auto center1 = point3(400, 400, 200);
  auto center2 = center1 + vec3(30, 0, 0);
  auto moving_sphere_material = new lambertian(color(0.7, 0.3, 0.1));
  ret[index++] =
      new moving_sphere(center1, center2, 0, 1, 50, moving_sphere_material);

  ret[index++] = new sphere(point3(260, 150, 45), 50, new dielectric(1.5));
  ret[index++] =
      new sphere(point3(0, 150, 145), 50, new metal(color(0.8, 0.8, 0.9), 1.0));

  // constant medium
  auto sphere_dielectric_2 =
      new sphere(point3(360, 150, 145), 70, new dielectric(1.5));
  ret[index++] = sphere_dielectric_2;
  ret[index++] =
      new constant_medium(sphere_dielectric_2, 0.2, color(0.2, 0.4, 0.9));

  // fog
  auto fog = new sphere(point3(0, 0, 0), 5000, new dielectric(1.5));
  ret[index++] = new constant_medium(fog, 0.0001, color(1, 1, 1));

  // earth
  auto earth_texture = new image_texture(data, w, h);
  auto earth_surface = new lambertian(earth_texture);
  ret[index++] = new sphere(point3(400, 200, 400), 100, earth_surface);

  auto pertext = new noise_texture(0.1, local_rand_state);
  ret[index++] = new sphere(point3(220, 280, 300), 80, new lambertian(pertext));

  auto white = new lambertian(color(.73, .73, .73));

  const int ns = 1000;
  hittable *cluster[ns];
  for (int j = 0; j < ns; j++) {
    cluster[j] = new sphere(random_vec3(0, 165, local_rand_state), 10, white);
  }

  ret[index++] = new translate(
      new rotate_y(new bvh_node(cluster, 0, ns, 0, 1, local_rand_state), 15),
      vec3(-100, 270, 395));

  return new bvh_node(ret, 0, index, 0.0, 1.0, local_rand_state);
}

__device__ hittable *simple_triangle(curandState *local_rand_state) {
  hittable *ret[3];
  int index = 0;
  // light
  auto light = new diffuse_light(color(17, 17, 17));
  ret[index++] = new xz_rect(123, 423, 147, 412, 554, light);

  auto white = new lambertian(color(.073, .73, .73));
  ret[index++] = new triangle(vec3(123, 0, 150), vec3(423, 0, 150),
                              vec3(273, 50, (500 + 150) / 2), vec3(0, 1, 0),
                              vec3(0, 1, 0), vec3(0, 1, 0), white);

  ret[index++] = new sphere(point3(273, 100, (500 + 150) / 2), 10,
                            new lambertian(color(0.5, 0.5, 0.5)));

  return new bvh_node(ret, 0, index, 0, 1, local_rand_state);
}

__device__ hittable *obj_model(triangle *tri_data, int tri_sz,
                               curandState *local_rand_state) {
  hittable **ret = new hittable *[tri_sz + 13];

  auto red = new lambertian(color(.65, .05, .05));
  auto white = new lambertian(color(.73, .73, .73));
  auto green = new lambertian(color(.12, .45, .15));

  int index = 0;

  auto blue_1 = new lambertian(color(0, 129.0f / 256.0, 167.0f / 256.0));

  // 0, 175, 185
  auto blue_2 = new lambertian(color(0, 175.0f / 256.0, 185.0f / 256.0));

  auto red_1 =
      new lambertian(color(240.0f / 256.0, 113.0f / 256.0, 103.0f / 256.0));

  // 253, 252, 220
  auto yellow_1 =
      new lambertian(color(253.0f / 256.0, 252.0f / 256.0, 220.0f / 256.0));

  // 254, 217, 183
  auto yellow_2 =
      new lambertian(color(254.0f / 256.0, 217.0f / 256.0, 183.0f / 256.0));

  auto back_metal = new metal(color(0.8, 0.8, 0.9), 0.01);

  auto light = new diffuse_light(color(20, 20, 20));

  ret[index++] = new sphere(point3(-1, 3.69 + 1, -2.5), 0.3, light);

  ret[index++] = new sphere(point3(1, 3.69 + 1, -2.5), 0.3,
                            new diffuse_light(color(20, 20, 10)));

  ret[index++] = new xz_rect(-4, 4, 1, 2, 4 + 1 - 0.01, light);
  ret[index++] = new xz_rect(-4, 4, 1, 2, -4 + 0.01, light);

  // fog
  // auto fog = new sphere(point3(0, 0, 0), 10, new dielectric(1.5));
  // ret[index++] = new constant_medium(fog, 0.0001, color(1, 1, 1));

  // glass
  // ret[index++] = new sphere(point3(2, 2, 1.5), 0.75, new dielectric(1.5));
  // ret[index++] = new sphere(point3(0, 0, 2), 0.5,
  //                           new metal(color(0.8, 0.8, 0.9), 0.0001));

  // back
  ret[index++] = new xy_rect(-4, 4, -4, 4 + 1, -4, blue_2);

  // bottom
  ret[index++] = new xz_rect(-40, 40, -40, 40, -4, red_1);

  // top
  ret[index++] = new xz_rect(-40, 40, -40, 40, 4 + 1, blue_1);

  // left and right
  ret[index++] = new yz_rect(-4, 4 + 1, -4, 4, -4, yellow_2);
  ret[index++] = new yz_rect(-4, 4 + 1, -4, 4, 4, yellow_2);
  ret[index++] =
      new yz_rect(-1, 3 + 1, -4, 4, -4, new metal(color(0.8, 0.8, 0.9), 0.01));
  ret[index++] = new yz_rect(-1, 3 + 1 - 0.001, -4, 4, 3.999,
                             new metal(color(0.8, 0.8, 0.9), 0.01));

  vec3 v1(-1, 1, 1);
  vec3 v2(-1, -1, 1);
  vec3 v3(-1, 1, -1);
  vec3 v4(-1, -1, -1);
  vec3 v5(1, 1, 1);
  vec3 v6(1, -1, 1);
  vec3 v7(1, 1, -1);
  vec3 v8(1, -1, -1);

  vec3 vn1(0, 1, 0);
  vec3 vn2(0, 0, -1);
  vec3 vn3(1, 0, 0);
  vec3 vn4(0, -1, 0);
  vec3 vn5(-1, 0, 0);
  vec3 vn6(0, 0, 1);

  // ret[index++] = new triangle(v3, v7, v5, vn1, vn1, vn1, green);
  // ret[index++] = new triangle(v1, v3, v5, vn1, vn1, vn1, red);

  // ret[index++] = new triangle(v8, v6, v7, vn3, vn3, vn3, green);

  // ret[index++] = new triangle(v6, v5, v7, vn3, vn3, vn3, red);

  // ret[index++] = new triangle(v6, v2, v5, vn6, vn6, vn6, red);

  // ret[index++] = new triangle(v2, v1, v5, vn6, vn6, vn6, green);

  // comment out this line the vertical square disappear
  // the problem seems to be inside bvh algo.

  for (int i = 0; i < tri_sz; i++) {
    // ((triangle *)tri_ptr[i])->mat_ptr = red;
    // triangle *tri = (triangle *)tri_ptr[i];
    // printf("%f, %f, %f\n", tri_data[i].v0.x(), tri_data[i].v0.y(),
    //        tri_data[i].v0.z());

    // ret[index++] = new triangle(tri_data[i].v0, tri_data[i].v1,
    //    tri_data[i].v2, tri_data[i].vn0, tri_data[i].vn1,
    //                             tri_data[i].vn2, index % 2 == 0 ? red :
    //                             green);

    // ret[index++] = new triangle(
    //     vec3(tri_data[i].v0.x(), tri_data[i].v0.y(), tri_data[i].v0.z()),
    //     vec3(tri_data[i].v1.x(), tri_data[i].v1.y(), tri_data[i].v1.z()),
    //     vec3(tri_data[i].v2.x(), tri_data[i].v2.y(), tri_data[i].v2.z()),
    //     tri_data[i].vn0, tri_data[i].vn1, tri_data[i].vn2,
    //     index % 2 == 0 ? red : green);

    // ret[index++] = new triangle(
    //     vec3(tri_data[i].v0.x(), tri_data[i].v0.y(), tri_data[i].v0.z()),
    //     vec3(tri_data[i].v1.x(), tri_data[i].v1.y(), tri_data[i].v1.z()),
    //     vec3(tri_data[i].v2.x(), tri_data[i].v2.y(), tri_data[i].v2.z()),
    //     tri_data[i].vn0, tri_data[i].vn1, tri_data[i].vn2, white);

    ret[index++] = new rotate_y(
        new triangle(
            vec3(tri_data[i].v0.x(), tri_data[i].v0.y(), tri_data[i].v0.z()),
            vec3(tri_data[i].v1.x(), tri_data[i].v1.y(), tri_data[i].v1.z()),
            vec3(tri_data[i].v2.x(), tri_data[i].v2.y(), tri_data[i].v2.z()),
            tri_data[i].vn0, tri_data[i].vn1, tri_data[i].vn2, white),
        30);
  }

  return new bvh_node(ret, 0, index, 0.0, 1.0, local_rand_state);
}

__global__ void create_world(hittable **d_list, hittable **d_world,
                             camera **d_camera, int nx, int ny,
                             curandState *rand_state, unsigned char *data,
                             int w, int h, color **background,
                             triangle *tri_data, int tri_sz) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {

    curandState *local_rand_state = rand_state;

    vec3 lookfrom(13, 2, 3);
    vec3 lookat(0, 0, 0);
    // float dist_to_focus = (lookfrom - lookat).length();
    float aperture = 0.00;
    float vfov = 40.0;
    vec3 vup(0, 1, 0);
    // background = new color(0, 0, 0);

    switch (0) {
    case 1:
      *d_world = random_scene(d_list, local_rand_state);
      vfov = 20.0;
      aperture = 0.05;
      *background = new color(0.70, 0.80, 1.00);
      break;

    case 2:
      *d_world = two_spheres(local_rand_state);
      vfov = 20.0;
      aperture = 0;
      *background = new color(0.70, 0.80, 1.00);
      break;

    case 3:
      *d_world = two_perlin_spheres(local_rand_state);
      vfov = 20.0;
      aperture = 0;
      *background = new color(0.70, 0.80, 1.00);
      break;
    case 4:
      *d_world = earth(data, w, h, local_rand_state);
      *background = new color(0.70, 0.80, 1.00);
      break;

    case 5:
      *background = new color(0.0, 0.0, 0.0);
      *d_world = simple_light(local_rand_state);
      lookfrom = point3(26, 3, 6);
      lookat = point3(0, 2, 0);
      vfov = 20.0f;
      break;

    case 6:
      *background = new color(0.0, 0.0, 0.0);
      // *background = new color(0.70, 0.80, 1.00);
      *d_world = cornell_box(local_rand_state);
      lookfrom = point3(278, 278, -800);
      lookat = point3(278, 278, 0);
      vfov = 40.0;
      break;

    case 7:
      *background = new color(0.0, 0.0, 0.0);
      *d_world = cornell_smoke(local_rand_state);
      lookfrom = point3(278, 278, -800);
      lookat = point3(278, 278, 0);
      vfov = 40.0;
      break;

    case 8:
      *background = new color(0.0, 0.0, 0.0);
      *d_world = rt_next_week_final_scene(data, w, h, local_rand_state);
      lookfrom = point3(478, 278, -600);
      lookat = point3(278, 278, 0);
      vfov = 40.0;
      break;

    case 9:
      // *background = new color(0.70/2, 0.80/2, 1.00/2);
      *background = new color(0.0, 0.0, 0.0);
      *d_world = simple_triangle(local_rand_state);
      lookfrom = point3(278, 278, -600);
      lookat = point3(278, 278, 0);
      vfov = 50.0;
      break;
    default:
    case 10:
      // *background = new color(0.70 / 2, 0.80 / 2, 1.00 / 2);
      *background = new color(0.0, 0.0, 0.0);
      *d_world = obj_model(tri_data, tri_sz, local_rand_state);
      lookfrom = point3(1, 3, 7);
      lookat = point3(0, 2, 0);
      vfov = 60.0;
      break;
    }

    float dist_to_focus = (lookfrom - lookat).length();
    *d_camera = new camera(lookfrom, lookat, vup, vfov, float(nx) / float(ny),
                           aperture, dist_to_focus, 0.0f, 1.0f);
    rand_state = local_rand_state;
  }
}

__global__ void free_world(hittable **d_list, hittable **d_world,
                           camera **d_camera) {
  for (int i = 0; i < 22 * 22 + 1 + 3; i++) {
    // the bug is located here, we have sphere and moving_sphere, but we only
    // use sphere here, a workaround is define moving_sphere as a sub class of
    // sphere. then we can get ride of cudaFree 700 error. delete ((sphere
    // *)d_list[i])->mat_ptr;
    delete d_list[i];
  }
  delete *d_world;
  delete *d_camera;
}

__global__ void set_triangle(triangle *tri_data, hittable **tri_ptr,
                             int tri_data_size, int max_x, int max_y) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;
  if ((i >= max_x) || (j >= max_y))
    return;
  int index = j * max_x + i;
  // printf("%d %d\n", index, tri_data_size);
  if (index < tri_data_size) {
    // printf("%d\n", index);
    tri_data[index].mat_ptr = new lambertian(color(.073, .73, .73));
    // new diffuse_light(color(15, 15, 15));
    tri_ptr[index] =
        new triangle(tri_data[index].v0, tri_data[index].v1, tri_data[index].v2,
                     tri_data[index].vn0, tri_data[index].vn1,
                     tri_data[index].vn2, new lambertian(color(.73, .73, .73)));
    // tri_ptr[index] =
    //     new triangle(tri_data[index].v0, tri_data[index].v1,
    //     tri_data[index].v2,
    //                  tri_data[index].vn0, tri_data[index].vn1,
    //                  tri_data[index].vn2, new diffuse_light(color(15, 15,
    //                  15)));
  }
  // printf("\n");
}

int main() {
  cudaDeviceSetLimit(cudaLimitStackSize, 32768ULL);

  size_t p;
  cuCtxGetLimit(&p, CU_LIMIT_MALLOC_HEAP_SIZE);
  std::cerr << p << std::endl;

  cudaDeviceSetLimit(cudaLimitMallocHeapSize, 512ULL * 1024ULL * 1024ULL);
  cuCtxGetLimit(&p, CU_LIMIT_MALLOC_HEAP_SIZE);
  std::cerr << p << std::endl;
  /**
   *    read obj file
   */

  // read data from file and generate a vector of triangles
  std::vector<triangle> triangles;
  read_triangles(triangles);

  // allocate memory in device memory
  triangle *tri_data;
  int tri_sz = triangles.size();

  checkCudaErrors(
      cudaMalloc((void **)&tri_data, triangles.size() * sizeof(triangle)));

  checkCudaErrors(cudaDeviceSynchronize());

  // copy the triangles to the device memory
  checkCudaErrors(cudaMemcpy((void *)tri_data, (void *)triangles.data(),
                             tri_sz * sizeof(triangle),
                             cudaMemcpyHostToDevice));
  checkCudaErrors(cudaDeviceSynchronize());

  // this is like d_list
  hittable **tri_data_ptr;
  checkCudaErrors(
      cudaMalloc((void **)&tri_data_ptr, tri_sz * sizeof(hittable *)));

  checkCudaErrors(cudaDeviceSynchronize());

  // initialize **tri_data_ptr
  int dnx = 128;
  int dny = tri_sz / 128 + 1;

  dim3 dblocks(dnx / 8 + 1, dny / 8 + 1);
  dim3 dthreads(8, 8);

  // set_triangle<<<dblocks, dthreads>>>(tri_data, tri_data_ptr, tri_sz, dnx,
  // dny); checkCudaErrors(cudaGetLastError());
  // checkCudaErrors(cudaDeviceSynchronize());

  /**
   *    read earthmap
   */
  const char *filename = "earthmap.jpeg";

  int width, height;
  int components_per_pixel = image_texture::bytes_per_pixel;

  unsigned char *data;

  data = stbi_load(filename, &width, &height, &components_per_pixel,
                   components_per_pixel);

  unsigned char *device_data;

  size_t img_data_size =
      components_per_pixel * width * height * sizeof(unsigned char);
  checkCudaErrors(cudaMallocManaged((void **)&device_data, img_data_size));

  checkCudaErrors(cudaMemcpy((void *)device_data, (void *)data, img_data_size,
                             cudaMemcpyHostToDevice));

  color **background_color;
  checkCudaErrors(
      cudaMallocManaged((void **)&background_color, sizeof(color *)));

  /**
   *    setup picture information
   */

  const auto aspect_ratio = 1.0; // 3.0 / 2.0;
  int nx = 800;                  // 1200;
  int ny = static_cast<int>(nx / aspect_ratio);
  // int ns = 10000; // 500*4; // 500;
  int ns = 50;
  int tx = 8;
  int ty = 8;

  std::cerr << "Rendering a " << nx << "x" << ny << " image with " << ns
            << " samples per pixel ";
  std::cerr << "in " << tx << "x" << ty << " blocks.\n";

  int num_pixels = nx * ny;
  size_t fb_size = num_pixels * sizeof(vec3);

  // allocate FB
  vec3 *fb;
  checkCudaErrors(cudaMallocManaged((void **)&fb, fb_size));

  // allocate random state
  curandState *d_rand_state;
  checkCudaErrors(
      cudaMalloc((void **)&d_rand_state, num_pixels * sizeof(curandState)));
  curandState *d_rand_state2;
  checkCudaErrors(cudaMalloc((void **)&d_rand_state2, 1 * sizeof(curandState)));

  // we need that 2nd random state to be initialized for the world creation
  rand_init<<<1, 1>>>(d_rand_state2);
  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaDeviceSynchronize());

  // make our world of hitables & the camera
  hittable **d_list;
  int num_hitables = 22 * 22 + 1 + 3;

  checkCudaErrors(
      cudaMalloc((void **)&d_list, num_hitables * sizeof(hittable *)));

  hittable **d_world;
  // checkCudaErrors(cudaMalloc((void **)&d_world, sizeof(hittable *)));
  checkCudaErrors(cudaMalloc((void **)&d_world, sizeof(bvh_node *)));
  camera **d_camera;
  checkCudaErrors(cudaMalloc((void **)&d_camera, sizeof(camera *)));

  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaDeviceSynchronize());

  std::cerr << "create world\n";
  clock_t start, stop;
  start = clock();

  create_world<<<1, 1>>>(d_list, d_world, d_camera, nx, ny, d_rand_state2,
                         device_data, width, height, background_color, tri_data,
                         tri_sz);
  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaDeviceSynchronize());

  stop = clock();
  double timer_seconds = ((double)(stop - start)) / CLOCKS_PER_SEC;
  std::cerr << "took " << timer_seconds << " seconds.\n";

  start = clock();
  // Render our buffer
  dim3 blocks(nx / tx + 1, ny / ty + 1);
  dim3 threads(tx, ty);

  render_init<<<blocks, threads>>>(nx, ny, d_rand_state);

  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaDeviceSynchronize());

  std::cerr << "start render\n";

  render<<<blocks, threads>>>(fb, nx, ny, ns, d_camera, d_world, d_rand_state,
                              background_color);
  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaDeviceSynchronize());

  stop = clock();
  timer_seconds = ((double)(stop - start)) / CLOCKS_PER_SEC;
  std::cerr << "took " << timer_seconds << " seconds.\n";

  // Output FB as Image
  std::cout << "P3\n" << nx << " " << ny << "\n255\n";
  for (int j = ny - 1; j >= 0; j--) {
    for (int i = 0; i < nx; i++) {
      size_t pixel_index = j * nx + i;
      int ir = int(255.99 * fb[pixel_index].x());
      int ig = int(255.99 * fb[pixel_index].y());
      int ib = int(255.99 * fb[pixel_index].z());
      std::cout << ir << " " << ig << " " << ib << "\n";
    }
  }

  // clean up
  checkCudaErrors(cudaDeviceSynchronize());
  free_world<<<1, 1>>>(d_list, d_world, d_camera);
  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaFree(d_camera));
  checkCudaErrors(cudaFree(d_world));
  checkCudaErrors(cudaFree(d_list));
  checkCudaErrors(cudaFree(d_rand_state));
  checkCudaErrors(cudaFree(d_rand_state2));
  checkCudaErrors(cudaFree(fb));

  cudaDeviceReset();

  return 0;
}
