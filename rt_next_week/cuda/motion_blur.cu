#include "aarect.h"
#include "bvh.h"
#include "camera.h"
#include "cuda_utils.h"
#include "hittable_list.h"
#include "material.h"
#include "moving_sphere.h"
#include "ray.h"
#include "sphere.h"
#include "vec3.h"
#include <curand_kernel.h>
#include <float.h>
#include <iostream>
#include <time.h>

// Matching the C++ code would recurse enough into color() calls that
// it was blowing up the stack, so we have to turn this into a
// limited-depth loop instead.  Later code in the book limits to a max
// depth of 50, so we adapt this a few chapters early on the GPU.
__device__ vec3 get_color(const ray &r, hittable **world,
                          curandState *local_rand_state) {
  ray cur_ray = r;
  vec3 cur_attenuation(1.0f, 1.0f, 1.0f);
  for (int i = 0; i < 50; i++) {
    hit_record rec;
    if ((*world)->hit(cur_ray, 0.001f, FLT_MAX, rec)) {
      ray scattered;
      vec3 attenuation;
      if (rec.mat_ptr->scatter(cur_ray, rec, attenuation, scattered,
                               local_rand_state)) {
        cur_attenuation *= attenuation;
        cur_ray = scattered;
      } else {
        return vec3(0.0, 0.0, 0.0);
      }
    } else {
      vec3 unit_direction = unit_vector(cur_ray.direction());
      float t = 0.5f * (unit_direction.y() + 1.0f);
      vec3 c = (1.0f - t) * vec3(1.0, 1.0, 1.0) + t * vec3(0.5, 0.7, 1.0);
      return cur_attenuation * c;
    }
  }
  return vec3(0.0, 0.0, 0.0); // exceeded recursion
}

__device__ vec3 get_color(const ray &r, color **background, hittable **world,
                          curandState *local_rand_state) {
  ray cur_ray = r;
  vec3 cur_attenuation(1.0f, 1.0f, 1.0f);

  const int depth = 50;

  vec3 emitted_rec[depth];
  vec3 attenuation_rec[depth];

  for (int i = 0; i < depth; i++) {
    hit_record rec;
    if ((*world)->hit(cur_ray, 0.001f, FLT_MAX, rec)) {

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
      // no hit
      // only have background
      cur_attenuation *= **background;
      while (i-- > 0) {
        cur_attenuation = emitted_rec[i] + cur_attenuation * attenuation_rec[i];
      }

      return cur_attenuation;
    }
  }
  return **background; // exceeded recursion
}

__global__ void rand_init(curandState *rand_state) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    curand_init(1984, 0, 0, rand_state);
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
  curand_init(1984 + pixel_index, 0, 0, &rand_state[pixel_index]);
}

__global__ void render(vec3 *fb, int max_x, int max_y, int ns, camera **cam,
                       hittable **world, curandState *rand_state,
                       color **background) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;
  if ((i >= max_x) || (j >= max_y))
    return;
  int pixel_index = j * max_x + i;
  curandState local_rand_state = rand_state[pixel_index];
  vec3 col(0, 0, 0);
  for (int s = 0; s < ns; s++) {
    float u = float(i + curand_uniform(&local_rand_state)) / float(max_x);
    float v = float(j + curand_uniform(&local_rand_state)) / float(max_y);
    ray r = (*cam)->get_ray(u, v, &local_rand_state);
    col += get_color(r, background, world, &local_rand_state);
  }
  rand_state[pixel_index] = local_rand_state;
  col /= float(ns);
  col[0] = sqrt(col[0]);
  col[1] = sqrt(col[1]);
  col[2] = sqrt(col[2]);
  fb[pixel_index] = col;
}

#define RND (curand_uniform(&local_rand_state))

__device__ hittable *random_scene(hittable **d_list,
                                  curandState local_rand_state) {
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

  return new bvh_node(d_list, 0, 22 * 22 + 1 + 3, 0.0f, 1.0f,
                      &local_rand_state);
}

__device__ hittable *two_spheres(curandState local_rand_state) {

  auto checker =
      new checker_texture(color(0.2, 0.3, 0.1), color(0.9, 0.9, 0.9));

  hittable *ret[2];
  ret[0] = new sphere(point3(0, -10, 0), 10, new lambertian(checker));
  ret[1] = new sphere(point3(0, 10, 0), 10, new lambertian(checker));

  return new bvh_node(ret, 0, 2, 0.0f, 1.0f, &local_rand_state);
}

__device__ hittable *two_perlin_spheres(curandState local_rand_state) {

  auto perlin_texture = new noise_texture(4, &local_rand_state);

  hittable *ret[2];
  ret[0] =
      new sphere(point3(0, -1000, 0), 1000, new lambertian(perlin_texture));
  ret[1] = new sphere(point3(0, 2, 0), 2, new lambertian(perlin_texture));

  return new bvh_node(ret, 0, 2, 0.0f, 1.0f, &local_rand_state);
}

__device__ hittable *earth(unsigned char *data, int w, int h,
                           curandState local_rand_state) {
  auto earth_texture = new image_texture(data, w, h);
  auto earth_surface = new lambertian(earth_texture);

  hittable *ret[1];
  ret[0] = new sphere(point3(0, 0, 0), 2, earth_surface);
  return new bvh_node(ret, 0, 1, 0.0f, 1.0f, &local_rand_state);
}

__device__ hittable *simple_light(curandState local_rand_state) {
  auto perlin_texture = new noise_texture(4, &local_rand_state);

  hittable *ret[4];
  ret[0] =
      new sphere(point3(0, -1000, 0), 1000, new lambertian(perlin_texture));
  ret[1] = new sphere(point3(0, 2, 0), 2, new lambertian(perlin_texture));

  auto diff_light = new diffuse_light(color(4, 4, 4));

  ret[2] = new xy_rect(3, 5, 1, 2, -2, diff_light);

  auto diff_light2 = new diffuse_light(color(6, 4, 4));
  ret[3] = new sphere(point3(0, 6, 0), 1.5, diff_light2);

  return new bvh_node(ret, 0, 4, 0.0f, 1.0f, &local_rand_state);
}

__device__ hittable *cornell_box(curandState local_rand_state) {
  hittable *ret[6];
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

  return new bvh_node(ret, 0, 6, 0.0f, 1.0f, &local_rand_state);
}

__global__ void create_world(hittable **d_list, hittable **d_world,
                             camera **d_camera, int nx, int ny,
                             curandState *rand_state, unsigned char *data,
                             int w, int h, color **background) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {

    curandState local_rand_state = *rand_state;

    vec3 lookfrom(13, 2, 3);
    vec3 lookat(0, 0, 0);
    // float dist_to_focus = (lookfrom - lookat).length();
    float aperture = 0.05;
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
    default:
    case 6:
      *background = new color(0.0, 0.0, 0.0);
      // *background = new color(0.70, 0.80, 1.00);
      *d_world = cornell_box(local_rand_state);
      lookfrom = point3(278, 278, -800);
      lookat = point3(278, 278, 0);
      vfov = 40.0;
      break;
    }

    float dist_to_focus = (lookfrom - lookat).length();
    *d_camera = new camera(lookfrom, lookat, vup, vfov, float(nx) / float(ny),
                           aperture, dist_to_focus, 0.0f, 1.0f);
    *rand_state = local_rand_state;
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

int main() {
  cudaDeviceSetLimit(cudaLimitStackSize, 32768ULL);

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

  const auto aspect_ratio = 3.0 / 2.0;
  int nx = 1200;
  int ny = static_cast<int>(nx / aspect_ratio);
  int ns = 500;
  //   int ns = 500;
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

  create_world<<<1, 1>>>(d_list, d_world, d_camera, nx, ny, d_rand_state2,
                         device_data, width, height, background_color);
  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaDeviceSynchronize());

  clock_t start, stop;
  start = clock();
  // Render our buffer
  dim3 blocks(nx / tx + 1, ny / ty + 1);
  dim3 threads(tx, ty);
  render_init<<<blocks, threads>>>(nx, ny, d_rand_state);
  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaDeviceSynchronize());
  render<<<blocks, threads>>>(fb, nx, ny, ns, d_camera, d_world, d_rand_state,
                              background_color);
  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaDeviceSynchronize());

  stop = clock();
  double timer_seconds = ((double)(stop - start)) / CLOCKS_PER_SEC;
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
}
