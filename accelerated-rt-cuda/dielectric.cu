#include "camera.h"
#include "color.h"
#include "cuda_utils.h"
#include "hittable_list.h"
#include "material.h"
#include "ray.h"
#include "sphere.h"
#include "vec3.h"
#include <curand_kernel.h>
#include <float.h> // for FLT_MAX
#include <iostream>

// this is a kernel function used to init the rand state for all pixels
__global__ void render_init(int max_x, int max_y, curandState *rand_state) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;
  if ((i >= max_x) || (j >= max_y))
    return;
  int pixel_index = j * max_x + i;

  // Each thread gets same seed, a different sequence number ,no offset
  curand_init(1984, pixel_index, 0, &rand_state[pixel_index]);
}

__device__ vec3 ray_color(ray r, hittable **world) {
  hit_record rec;
  if ((*world)->hit(r, 0.0, FLT_MAX, rec)) {
    return 0.5f * vec3(rec.normal.x() + 1.0f, rec.normal.y() + 1.0f,
                       rec.normal.z() + 1.0f);
  } else {
    vec3 unit_direction = unit_vector(r.direction());
    float t = 0.5f * (unit_direction.y() + 1.0f);
    return (1.0f - t) * vec3(1.0, 1.0, 1.0) + t * vec3(0.5, 0.7, 1.0);
  }
}

// for diffuse effect
// with material
__device__ vec3 ray_color(ray r, hittable **world,
                          curandState *local_rand_state) {
  ray cur_ray = r;
  vec3 cur_attenuation = vec3(1.0, 1.0, 1.0);
  // bounce 50 times
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
        // the material will absorb all light ?
        return vec3(0.0, 0.0, 0.0);
      }

    } else {
      // shoot into the sky, a light source
      vec3 unit_direction = unit_vector(cur_ray.direction());
      float t = 0.5f * (unit_direction.y() + 1.0f);
      vec3 c = (1.0f - t) * vec3(1.0, 1.0, 1.0) + t * vec3(0.5, 0.7, 1.0);
      // modify the intensity
      return cur_attenuation * c;
    }
  }
  return vec3(0.0, 0.0, 0.0); // exceeded recursion
}

__global__ void render(vec3 *fb, int max_x, int max_y, int ns, camera **cam,
                       hittable **world, curandState *rand_state) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;
  if ((i >= max_x) || (j >= max_y))
    return;
  //   int pixel_index = j*max_x*3 + i*3;
  int pixel_index = j * max_x + i;
  curandState local_rand_state = rand_state[pixel_index];
  vec3 col(0, 0, 0);

  for (int s = 0; s < ns; s++) {
    float u = float(i + curand_uniform(&local_rand_state)) / float(max_x);
    float v = float(j + curand_uniform(&local_rand_state)) / float(max_y);
    ray r = (*cam)->get_ray(u, v, &local_rand_state);
    // sample ns times, each has a random direction
    col += ray_color(r, world, &local_rand_state);
  }

  rand_state[pixel_index] = local_rand_state;

  col /= float(ns);
  col[0] = sqrt(col[0]);
  col[1] = sqrt(col[1]);
  col[2] = sqrt(col[2]);
  fb[pixel_index] = col;
}

__global__ void create_world(hittable **d_list, hittable **d_world,
                             camera **cam, int nx, int ny) {
  // make sure only create this object once
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    d_list[0] =
        new sphere(vec3(0, 0, -1), 0.5, new lambertian(vec3(0.1, 0.2, 0.5)));
    d_list[1] = new sphere(vec3(0, -100.5, -1), 100,
                           new lambertian(vec3(0.8, 0.8, 0.0)));
    d_list[2] =
        new sphere(vec3(1, 0, -1), 0.5, new metal(vec3(0.8, 0.6, 0.2), 0.0));
    d_list[3] = new sphere(vec3(-1, 0, -1), 0.5, new dielectric(1.5));
    d_list[4] = new sphere(vec3(-1, 0, -1), -0.45, new dielectric(1.5));
    *d_world = new hittable_list(d_list, 5);
    // *(cam) = new camera();
    // *cam = new camera(vec3(-2, 2, 1), vec3(0, 0, -1), vec3(0, 1, 0), 20.0,
    //                   float(nx) / float(ny));
    vec3 lookfrom(3, 3, 2);
    vec3 lookat(0, 0, -1);
    float dist_to_focus = (lookfrom - lookat).length();
    float aperture = 2.0;
    *cam = new camera(lookfrom, lookat, vec3(0, 1, 0), 20.0f,
                      float(nx) / float(ny), aperture, dist_to_focus);
  }
}

__global__ void free_world(hittable **d_list, hittable **d_world,
                           camera **cam) {
  // note here we delete the inside pointer first as it is created by new
  for (int i = 0; i < 5; i++) {
    delete ((sphere *)d_list[i])->mat_ptr;
    delete d_list[i];
  }
  delete *d_world;
  delete *(cam);
}

int main(void) {
  // Image
  const auto aspect_ratio = 16.0 / 9.0;
  // nx = 1200 will cause an error in image viewer "pnm loader expected to find an integer"
  int nx = 1600;
  int ny = static_cast<int>(nx / aspect_ratio);
  // ny = 500;
  int num_pixels = nx * ny;
  int fb_size = num_pixels * sizeof(vec3);
  int ns = 100;

  // Camera

  //   auto viewport_height = 2.0;
  //   auto viewport_width = aspect_ratio * viewport_height;
  //   auto focal_length = 1.0;

  //   auto origin = point3(0, 0, 0);
  //   auto horizontal = vec3(viewport_width, 0, 0);
  //   auto vertical = vec3(0, viewport_height, 0);
  //   auto lower_left_corner =
  //       origin - horizontal / 2 - vertical / 2 - vec3(0, 0, focal_length);

  // a list of 2 hittable objects
  hittable **d_list;
  checkCudaErrors(cudaMalloc((void **)&d_list, 4 * sizeof(hittable *)));

  hittable **d_world;
  checkCudaErrors((cudaMalloc((void **)&d_world, sizeof(hittable *))));
  camera **d_cam;
  checkCudaErrors((cudaMalloc((void **)&d_cam, sizeof(camera *))));

  create_world<<<1, 1>>>(d_list, d_world, d_cam, nx, ny);
  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaDeviceSynchronize());

  // allocate FB, no initialization in CPU memory here.
  vec3 *fb;
  checkCudaErrors(cudaMallocManaged((void **)&fb, fb_size));

  // divide the work on GPU into blocks of 8x8 threads
  int tx = 8;
  int ty = 8;

  // Render our buffer
  dim3 blocks(nx / tx + 1, ny / ty + 1);
  dim3 threads(tx, ty);

  curandState *d_rand_state;
  checkCudaErrors(
      cudaMalloc((void **)&d_rand_state, num_pixels * sizeof(curandState)));
  render_init<<<blocks, threads>>>(nx, ny, d_rand_state);
  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaDeviceSynchronize());

  //   render<<<blocks, threads>>>(fb, nx, ny, lower_left_corner, horizontal,
  //                               vertical, origin, d_world);
  render<<<blocks, threads>>>(fb, nx, ny, ns, d_cam, d_world, d_rand_state);

  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaDeviceSynchronize());

  // Output FB as Image
  std::cout << "P3\n" << nx << " " << ny << "\n255\n";
  for (int j = ny - 1; j >= 0; j--) {
    for (int i = 0; i < nx; i++) {
      vec3 pixel = fb[j * nx + i];
      color pixel_color(pixel.x(), pixel.y(), pixel.z());
      write_color(std::cout, pixel_color);
    }
  }
  checkCudaErrors(cudaDeviceSynchronize());
  free_world<<<1, 1>>>(d_list, d_world, d_cam);

  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaFree(d_list));
  checkCudaErrors(cudaFree(d_world));
  checkCudaErrors(cudaFree(fb));
  checkCudaErrors(cudaFree(d_cam));

  return 0;
}