// #include "obj_parser.h"

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

int main() {
  // vector<triangle> triangles;

  std::string filename = "./objs/default_cube_in_triangles.obj";

  std::ifstream infile("./obj/test.obj");

  if (infile.is_open()) {
    std::string line;
    while (std::getline(infile, line)) {
      std::cout << line << std::endl;
    }
    infile.close();
  } else {
  
    std::cerr << "read failed" << std::endl;
  }
  return 0;
}