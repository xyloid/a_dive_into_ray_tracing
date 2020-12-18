#include "color.h"
#include "vec3.h"

#include "cuda_utils.h"
#include <iostream>

__global__ void render(float *fb, int max_x, int max_y) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;
  if((i >= max_x) || (j >= max_y)) return;
  int pixel_index = j*max_x*3 + i*3;
  fb[pixel_index + 0] = float(i) / max_x;
  fb[pixel_index + 1] = float(j) / max_y;
  fb[pixel_index + 2] = 0.2;
}


int main(void) {
  // Image
  const auto aspect_ratio = 3.0 / 2.0;
  int nx = 400;
  int ny = static_cast<int>(nx / aspect_ratio);
  int num_pixels = nx * ny;
  int fb_size = 3 * num_pixels * sizeof(float);

  // allocate FB, no initialization in CPU memory here.
  float *fb;
  checkCudaErrors(cudaMallocManaged((void **)&fb, fb_size));

  // divide the work on GPU into blocks of 8x8 threads
  int tx = 8;
  int ty = 8;

  // Render our buffer
  dim3 blocks(nx/tx+1,ny/ty+1);
  dim3 threads(tx,ty);
  render<<<blocks, threads>>>(fb, nx, ny);
  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaDeviceSynchronize());

  // Output FB as Image
  std::cout << "P3\n" << nx << " " << ny << "\n255\n";
  for (int j = ny-1; j >= 0; j--) {
      for (int i = 0; i < nx; i++) {
          size_t pixel_index = j*3*nx + i*3;
          float r = fb[pixel_index + 0];
          float g = fb[pixel_index + 1];
          float b = fb[pixel_index + 2];
          int ir = int(255.99*r);
          int ig = int(255.99*g);
          int ib = int(255.99*b);
          std::cout << ir << " " << ig << " " << ib << "\n";
      }
  }
  checkCudaErrors(cudaFree(fb));

  return 0;
}