#include "obj_parser.h"

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

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
        std::cout << "vn found " << std::endl
                  << line << std::endl
                  << x << "," << y << "," << z << std::endl;
      } else if (type == "v") {
        in >> x >> y >> z;
        vs.push_back(vec3(x, y, z));
        std::cout << "v found " << std::endl
                  << line << std::endl
                  << x << "," << y << "," << z << std::endl;
      } else if (type == "f") {
        // find face
        // format 1//1 2//2 3//2
        vector<int> indices;
        while (!in.eof()) {
          string section;
          in >> section;
          std::cout << "section: " << section << std::endl;
          char delimiter = '/';
          std::istringstream sec(section);
          string num;
          while (getline(sec, num, delimiter)) {
            if (num.length() == 0) {
              indices.push_back(-1);
            } else {
              float n = std::stof(num);
              std::cout << num << "\t" << n << std::endl;
              indices.push_back(--n);
            }
          }
        }
        triangles.push_back(triangle(vs.at(indices.at(0)), vs.at(indices.at(3)),
                                     vs.at(indices.at(6)), vs.at(indices.at(2)),
                                     vs.at(indices.at(5)), vs.at(indices.at(8)),
                                     nullptr));
      }

      // if (line.rfind("#", 0) == 0) {
      //   // found comments
      //   continue;
      // } else if (line.rfind("vn", 1) == 0) {
      //   // found normal vectors

      //   std::istringstream in(line.substr(1));
      //   in >> x >> y >> z;
      //   vns.push_back(vec3(x, y, z));
      //   std::cout << "vn found " << std::endl
      //             << line.substr(2) << std::endl
      //             << x << "," << y << "," << z << std::endl;
      // } else if (line.rfind("vt", 1) == 0) {
      //   // found texture data
      //   continue;
      // } else if (line.rfind("v", 0) == 0) {
      //   // found vertex
      //   std::cout << "v found" << std::endl;
      // } else if (line.rfind("f", 0) == 0) {
      //   // found faces
      //   std::cout << "face found" << std::endl;
      // }
    }
    infile.close();
  } else {

    std::cerr << "read failed" << std::endl;
  }

  for (vector<vec3>::iterator ptr = vs.begin(); ptr < vs.end(); ptr++) {
    std::cout << *ptr << std::endl;
  }
  return 0;
}