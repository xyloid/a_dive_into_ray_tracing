#include "camera.h"
#include "color.h"
#include "cuda_utils.h"
#include "hittable_list.h"
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

// function to generate a point inside the unit sphere
// this point is the direction of rays shooting randomly bc of diffuse.
__device__ vec3 random_in_unit_sphere(curandState *local_rand_state) {
  vec3 center = vec3(1, 1, 1);
  vec3 p;
  do {
    p = 2.0f * vec3(curand_uniform(local_rand_state),
                    curand_uniform(local_rand_state),
                    curand_uniform(local_rand_state)) -
        center;

  } while (p.length_squared() >= 1.0f);
  return p;
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
__device__ vec3 ray_color(ray r, hittable **world,
                          curandState *local_rand_state) {
  ray cur_ray = r;
  float cur_attenuation = 1.0f;
  // bounce 50 times
  for (int i = 0; i < 50; i++) {
    hit_record rec;
    if ((*world)->hit(cur_ray, 0.001f, FLT_MAX, rec)) {
      // hit, bounce into a random point in the unit sphere
      // apply unit_vector to get true lambertian reflection 
      vec3 target =
          rec.p + rec.normal + unit_vector(random_in_unit_sphere(local_rand_state));
      // intensity reduced after the hit -> change of direction
      cur_attenuation *= 0.5f;
      // current hit point and new direction.
      cur_ray = ray(rec.p, target - rec.p);
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
    ray r = (*cam)->get_ray(u, v);
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

__global__ void create_world(hittable **d_list, hittable **d_world) {
  // make sure only create this object once
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    *(d_list) = new sphere(vec3(0.0f, 0.0f, -1.0f), 0.5f);
    *(d_list + 1) = new sphere(vec3(0, -100.5, -1), 100);
    *d_world = new hittable_list(d_list, 2);
  }
}

__global__ void free_world(hittable **d_list, hittable **d_world) {
  delete *(d_list);
  delete *(d_list + 1);
  delete *d_world;
}

__global__ void create_camera(camera **cam) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    *(cam) = new camera();
  }
}

__global__ void free_camera(camera **cam) { delete *(cam); }

int main(void) {
  // Image
  const auto aspect_ratio = 16.0 / 9.0;
  int nx = 1200;
  int ny = static_cast<int>(nx / aspect_ratio);
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
  checkCudaErrors(cudaMalloc((void **)&d_list, 2 * sizeof(hittable *)));

  hittable **d_world;
  checkCudaErrors((cudaMalloc((void **)&d_world, sizeof(hittable *))));
  create_world<<<1, 1>>>(d_list, d_world);
  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaDeviceSynchronize());

  camera **d_cam;
  checkCudaErrors((cudaMalloc((void **)&d_cam, sizeof(camera *))));
  create_camera<<<1, 1>>>(d_cam);
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
  free_world<<<1, 1>>>(d_list, d_world);
  free_camera<<<1, 1>>>(d_cam);

  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaFree(d_list));
  checkCudaErrors(cudaFree(d_world));
  checkCudaErrors(cudaFree(fb));
  checkCudaErrors(cudaFree(d_cam));

  return 0;
}