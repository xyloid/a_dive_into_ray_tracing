#include "cuda_utils.h"
#include "material.h"
#include "obj_parser.h"
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

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

int main() {
  vector<vec3> vns;
  vector<vec3> vs;
  vector<triangle> triangles;

  // std::string filename = "objs/dafault_cube_in_triangles.obj";
  std::string filename = "objs/bunny.obj";

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
  cudaDeviceSetLimit(cudaLimitStackSize, 32768ULL);
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

  int nx = 128;
  int ny = size / 128 + 1;

  dim3 blocks(nx / tx + 1, ny / ty + 1);
  dim3 threads(tx, ty);

  set_triangle<<<blocks, threads>>>(tri_data, triangles.size(), nx, ny);

  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaDeviceSynchronize());

  checkCudaErrors(cudaFree(tri_data));

  return 0;
}