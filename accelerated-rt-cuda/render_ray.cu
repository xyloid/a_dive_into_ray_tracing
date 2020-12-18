#include "color.h"
#include "ray.h"
#include "vec3.h"

#include "cuda_utils.h"
#include <iostream>

__device__ vec3 ray_color(ray r) {
  vec3 unit_direction = unit_vector(r.direction());
  // force float instead of double during computation
  float t = 0.5f * (unit_direction.y() + 1.0f);
  return (1.0f - t) * vec3(1.0, 1.0, 1.0) + t * vec3(0.5, 0.7, 1.0);
}

__global__ void render(vec3 *fb, int max_x, int max_y, vec3 lower_left_corner,
                       vec3 horizontal, vec3 vertical, vec3 origin) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;
  if ((i >= max_x) || (j >= max_y))
    return;
  //   int pixel_index = j*max_x*3 + i*3;
  int pixel_index = j * max_x + i;
  float u = float(i) / float(max_x);
  float v = float(j) / float(max_y);
  ray r(origin, lower_left_corner + u * horizontal + v * vertical);
  fb[pixel_index] = ray_color(r);
}

int main(void) {
  // Image
  const auto aspect_ratio = 3.0 / 2.0;
  int nx = 400;
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

  // allocate FB, no initialization in CPU memory here.
  vec3 *fb;
  checkCudaErrors(cudaMallocManaged((void **)&fb, fb_size));

  // divide the work on GPU into blocks of 8x8 threads
  int tx = 8;
  int ty = 8;

  // Render our buffer
  dim3 blocks(nx / tx + 1, ny / ty + 1);
  dim3 threads(tx, ty);
  render<<<blocks, threads>>>(fb, nx, ny, lower_left_corner, horizontal,
                              vertical, origin);
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
  checkCudaErrors(cudaFree(fb));

  return 0;
}