#include <iostream>

#define checkCudaErrors(val) check_cuda((val), #val, __FILE__, __LINE__)
void check_cuda(cudaError_t result, char const *const func,
                const char *const file, int const line) {
  if (result) {
    std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at "
              << file << ":" << line << " '" << func << "' \n";
    cudaDeviceReset();
    exit(99);
  }
}

int main(void) {
  // Image
  const auto aspect_ratio = 3.0 / 2.0;
  int nx = 400;
  int ny = static_cast<int>(nx / aspect_ratio);
  int num_pixels = nx * ny;
  int fb_size = 3 * num_pixels * sizeof(float);

  // allocate FB
  float *fb;
  checkCudaErrors(cudaMallocManaged((void **)&fb, fb_size));

  // divide the work on GPU into blocks of 8x8 threads
  int tx = 8;
  int ty = 8;

  return 0;
}