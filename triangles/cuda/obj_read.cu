#include "bvh.h"
#include "camera.h"
#include "cuda_utils.h"
#include "material.h"
#include "obj_parser.h"
#include <curand_kernel.h>
#include <float.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

__device__ vec3 get_color(const ray &r, color **background, hittable **world,
                          curandState *local_rand_state) {
  ray cur_ray = r;
  vec3 cur_attenuation(1.0f, 1.0f, 1.0f);

  const int depth = 50;
  vec3 emitted_rec[depth];
  vec3 attenuation_rec[depth];

  for (int i = 0; i < depth; i++) {
    hit_record rec;
    // (*world)->hit(cur_ray, 0.001f, FLT_MAX, rec, local_rand_state);
    // hit_record rec;
    // if ((*world)->hit(cur_ray, 0.001f, FLT_MAX, rec, local_rand_state)) {

    //   ray scattered;
    //   vec3 attenuation;

    //   color emitted = rec.mat_ptr->emitted(rec.u, rec.v, rec.p);

    //   if (rec.mat_ptr->scatter(cur_ray, rec, attenuation, scattered,
    //                            local_rand_state)) {
    //     // scattered
    //     // cur_attenuation *= attenuation;
    //     // cur_attenuation += emitted;
    //     // cur_attenuation *= (attenuation + emitted);
    //     emitted_rec[i] = emitted;
    //     attenuation_rec[i] = attenuation;

    //     cur_ray = scattered;

    //   } else {
    //     // no scatter
    //     // no attenuation
    //     // no background light
    //     // but we have emitted

    //     cur_attenuation *= emitted;

    //     while (i-- > 0) {
    //       cur_attenuation =
    //           emitted_rec[i] + cur_attenuation * attenuation_rec[i];
    //     }

    //     return cur_attenuation;
    //   }
    // } else {
    //   // no hit
    //   // only have background
    //   cur_attenuation *= **background;
    //   while (i-- > 0) {
    //     cur_attenuation = emitted_rec[i] + cur_attenuation *
    //     attenuation_rec[i];
    //   }

    //   return cur_attenuation;
    // }
  }
  return **background; // exceeded recursion
}

__global__ void rand_init(curandState *rand_state) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    curand_init(1984, 0, 0, rand_state);
  }
}

// rand state for each pixel
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
  curandState local_rand_state = rand_state[pixel_index];
  printf("%p %p %d\n", &local_rand_state, &rand_state[pixel_index], pixel_index);
  vec3 col(0, 0, 0);
  for (int s = 0; s < ns; s++) {
    float u = float(i + curand_uniform(&local_rand_state)) / float(max_x);
    float v = float(j + curand_uniform(&local_rand_state)) / float(max_y);
    ray r = (*cam)->get_ray(u, v, &local_rand_state);
    // col += get_color(r, background, world, &local_rand_state);
  }
  // rand_state[pixel_index] = local_rand_state;
  // col /= float(ns);
  // col[0] = sqrt(col[0]);
  // col[1] = sqrt(col[1]);
  // col[2] = sqrt(col[2]);
  // fb[pixel_index] = col;
}

__global__ void set_triangle(triangle *tri_data, int tri_data_size, int max_x,
                             int max_y) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;
  if ((i >= max_x) || (j >= max_y))
    return;
  int index = j * max_x + i;
  // printf("%d %d\n", index, tri_data_size);
  if (index < tri_data_size) {
    // printf("%d\n", index);
    tri_data[index].mat_ptr = new lambertian(color(.73, .73, .73));
  }
  // printf("\n");
}

__global__ void create_world(hittable **d_list, hittable **d_world,
                             camera **d_camera, int nx, int ny,
                             curandState *rand_state, triangle *tri_data,
                             int tri_data_size, int w, int h,
                             color **background) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    curandState local_rand_state = *rand_state;

    vec3 lookfrom(3, 2, 3);
    vec3 lookat(0, 0, 0);
    // float dist_to_focus = (lookfrom - lookat).length();
    float aperture = 0.00;
    float vfov = 40.0;
    vec3 vup(0, 1, 0);

    *background = new color(0.70 / 2, 0.80 / 2, 1.00 / 2);
    // *background = new color(0, 0, 0);

    // page fault if we use this to allocate memory, I think the reason is large
    // amount of memory must be allocated before the execution start.
    // hittable **l = new hittable *[tri_data_size];

    for (int i = 0; i < tri_data_size; i++) {
      // printf("%d\n", i);
      // l[i] = (hittable *)&tri_data[i];
      // d_list[i] = (tri_data + i);
      d_list[i] = &tri_data[i];
    }

    *d_world =
        new bvh_node(d_list, 0, tri_data_size, 0.0, 1.0, &local_rand_state);

    float dist_to_focus = (lookfrom - lookat).length();
    *d_camera = new camera(lookfrom, lookat, vup, vfov, float(nx) / float(ny),
                           aperture, dist_to_focus, 0.0f, 1.0f);
    *rand_state = local_rand_state;
  }
}

int main() {
  vector<vec3> vns;
  vector<vec3> vs;
  vector<triangle> triangles;

  std::string filename = "objs/dafault_cube_in_triangles.obj";
  // std::string filename = "objs/bunny.obj";

  // std::ifstream infile("objs/test.obj");
  std::ifstream infile(filename);

  if (infile.is_open()) {
    std::string line;
    float x, y, z;
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
        // format 1//1 2//2 3//2
        vector<int> indices;
        while (!in.eof()) {
          string section;
          in >> section;
          // std::cout << "section: " << section << std::endl;
          char delimiter = '/';
          std::istringstream sec(section);
          string num;
          while (getline(sec, num, delimiter)) {
            if (num.length() == 0) {
              indices.push_back(-1);
            } else {
              float n = std::stof(num);
              // std::cout << num << "\t" << n << std::endl;
              indices.push_back(--n);
            }
          }
        }
        triangles.push_back(triangle(vs.at(indices.at(0)), vs.at(indices.at(3)),
                                     vs.at(indices.at(6)), vs.at(indices.at(2)),
                                     vs.at(indices.at(5)), vs.at(indices.at(8)),
                                     nullptr));
      }
    }
    infile.close();
  } else {

    std::cerr << "read failed" << std::endl;
  }

  // we have the triangles here
  cudaDeviceSetLimit(cudaLimitStackSize, 2*32768ULL);
  triangle *tri_data;

  checkCudaErrors(
      cudaMalloc((void **)&tri_data, triangles.size() * sizeof(triangle)));

  checkCudaErrors(cudaMemcpy((void *)tri_data, (void *)triangles.data(),
                             triangles.size() * sizeof(triangle),
                             cudaMemcpyHostToDevice));

  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaDeviceSynchronize());

  int size = triangles.size();
  int tx = 8;
  int ty = 8;

  int dnx = 128;
  int dny = size / 128 + 1;

  dim3 dblocks(dnx / tx + 1, dny / ty + 1);
  dim3 dthreads(tx, ty);

  set_triangle<<<dblocks, dthreads>>>(tri_data, triangles.size(), dnx, dny);

  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaDeviceSynchronize());

  const auto aspect_ratio = 1.0; // 3.0 / 2.0;
  int nx = 800;                  // 1200;
  int ny = static_cast<int>(nx / aspect_ratio);
  int ns = 50; // 500;

  int num_pixels = nx * ny;
  size_t fb_size = num_pixels * sizeof(vec3);

  // allocate FB
  vec3 *fb;
  checkCudaErrors(cudaMallocManaged((void **)&fb, fb_size));

  hittable **d_list;

  checkCudaErrors(cudaMalloc((void **)&d_list, size * sizeof(hittable *)));

  // allocate random state
  curandState *d_rand_state;
  checkCudaErrors(
      cudaMalloc((void **)&d_rand_state, num_pixels * sizeof(curandState)));
  curandState *d_rand_state2;
  checkCudaErrors(cudaMalloc((void **)&d_rand_state2, 1 * sizeof(curandState)));

  rand_init<<<1, 1>>>(d_rand_state2);
  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaDeviceSynchronize());

  hittable **d_world;
  // checkCudaErrors(cudaMalloc((void **)&d_world, sizeof(hittable *)));
  checkCudaErrors(cudaMalloc((void **)&d_world, sizeof(bvh_node *)));
  camera **d_camera;
  checkCudaErrors(cudaMalloc((void **)&d_camera, sizeof(camera *)));

  color **background_color;
  checkCudaErrors(
      cudaMallocManaged((void **)&background_color, sizeof(color *)));

  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaDeviceSynchronize());
  // create world

  create_world<<<1, 1>>>(d_list, d_world, d_camera, nx, ny, d_rand_state2,
                         tri_data, size, 0, 0, background_color);
  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaDeviceSynchronize());

  // render
  dim3 blocks(nx / tx + 1, ny / ty + 1);
  dim3 threads(tx, ty);

  render_init<<<blocks, threads>>>(nx, ny, d_rand_state);

  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaDeviceSynchronize());
  render<<<blocks, threads>>>(fb, nx, ny, ns, d_camera, d_world, d_rand_state,
                              background_color);
  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaDeviceSynchronize());

  checkCudaErrors(cudaFree(tri_data));

  return 0;
}