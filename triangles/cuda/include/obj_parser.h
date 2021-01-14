#ifndef OBJ_PARSER_H
#define OBJ_PARSER_H

#include <iostream>
#include <string>
#include <vector>

using std::string;
using std::vector;

#include "triangle.h"
#include "vec3.h"

class obj_parser {
public:
  obj_parser(string &_filename) : filename(_filename) {}

public:
  string filename;
  vector<vec3> Vs;
  vector<vec3> VNs;
  vector<triangle> triangle;

};

#endif