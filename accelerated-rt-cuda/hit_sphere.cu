#include "color.h"
#include "cuda_utils.h"
#include "hittable_list.h"
#include "ray.h"
#include "sphere.h"
#include "vec3.h"
#include <float.h> // for FLT_MAX
#include <iostream>

// __device__ float hit_sphere(const point3 &center, float radius, const ray &r)
// {
//   vec3 oc = r.origin() - center;
//   auto a = dot(r.direction(), r.direction());
//   auto b = 2.0f * dot(oc, r.direction());
//   auto c = dot(oc, oc) - radius * radius;
//   auto discriminant = b * b - 4.0f * a * c;
//   if (discriminant < 0.0f) {
//     return -1.0f;
//   } else {
//     return (-b - sqrt(discriminant)) / (2.0f * a);
//   }
// }

__device__ vec3 ray_color(ray r, hittable **world) {
  //   float t = hit_sphere(point3(0, 0, -1), 0.5f, r);
  //   if (t > 0.0f) {
  //     vec3 N = unit_vector(r.at(t) - vec3(0, 0, -1));
  //     return 0.5f * color(N.x() + 1, N.y() + 1, N.z() + 1);
  //   }
  //   vec3 unit_direction = unit_vector(r.direction());
  //   // force float instead of double during computation
  //   t = 0.5f * (unit_direction.y() + 1.0f);
  //   return (1.0f - t) * vec3(1.0, 1.0, 1.0) + t * vec3(0.5, 0.7, 1.0);
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

__global__ void render(vec3 *fb, int max_x, int max_y, vec3 lower_left_corner,
                       vec3 horizontal, vec3 vertical, vec3 origin,
                       hittable **world) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;
  if ((i >= max_x) || (j >= max_y))
    return;
  //   int pixel_index = j*max_x*3 + i*3;
  int pixel_index = j * max_x + i;
  float u = float(i) / float(max_x);
  float v = float(j) / float(max_y);
  ray r(origin, lower_left_corner + u * horizontal + v * vertical);
  fb[pixel_index] = ray_color(r, world);
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

int main(void) {
  // Image
  const auto aspect_ratio = 16.0 / 9.0;
  int nx = 1200;
  int ny = static_cast<int>(nx / aspect_ratio);
  int num_pixels = nx * ny;
  int fb_size = num_pixels * sizeof(vec3);

  // Camera

  auto viewport_height = 2.0;
  auto viewport_width = aspect_ratio * viewport_height;
  auto focal_length = 1.0;

  auto origin = point3(0, 0, 0);
  auto horizontal = vec3(viewport_width, 0, 0);
  auto vertical = vec3(0, viewport_height, 0);
  auto lower_left_corner =
      origin - horizontal / 2 - vertical / 2 - vec3(0, 0, focal_length);

  // a list of 2 hittable objects
  hittable **d_list;
  checkCudaErrors(cudaMalloc((void **)&d_list, 2 * sizeof(hittable *)));

  hittable **d_world;
  checkCudaErrors((cudaMalloc((void **)&d_world, sizeof(hittable *))));
  create_world<<<1, 1>>>(d_list, d_world);
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
  //   render<<<blocks, threads>>>(fb, nx, ny, lower_left_corner, horizontal,
  //                               vertical, origin, d_world);
  render<<<blocks, threads>>>(fb, nx, ny, vec3(-2.0, -1.0, -1.0),
                              vec3(4.0, 0.0, 0.0), vec3(0.0, 2.0, 0.0),
                              vec3(0.0, 0.0, 0.0), d_world);

  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaDeviceSynchronize());

  // Output FB as Image
  std::cout << "P3\n" << nx << " " << ny << "\n255\n";
  for (int j = ny - 1; j >= 0; j--) {
    for (int i = 0; i < nx; i++) {
      //   size_t pixel_index = j * 3 * nx + i * 3;
      //   float r = fb[pixel_index + 0];
      //   float g = fb[pixel_index + 1];
      //   float b = fb[pixel_index + 2];
      //   int ir = int(255.99 * r);
      //   int ig = int(255.99 * g);
      //   int ib = int(255.99 * b);

      vec3 pixel = fb[j * nx + i];
      color pixel_color(pixel.x(), pixel.y(), pixel.z());
      write_color(std::cout, pixel_color);
    }
  }
  checkCudaErrors(cudaDeviceSynchronize());
  free_world<<<1, 1>>>(d_list, d_world);
  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaFree(d_list));
  checkCudaErrors(cudaFree(d_world));
  checkCudaErrors(cudaFree(fb));

  return 0;
}