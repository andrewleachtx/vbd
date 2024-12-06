#include "utils.h"

#define _USE_MATH_DEFINES
#include <math.h>

#include "../../include.h"

#include <memory>
#include <iostream>
#include <vector>
#include <string>
#include <random>

using std::cout, std::endl, std::cerr;
using std::shared_ptr, std::make_shared;
using std::vector, std::string;

float randFloat() {
    return static_cast<float>(rand()) / RAND_MAX;
};

Eigen::Vector3f randXYZ() {
    return Eigen::Vector3f(randFloat(), randFloat(), randFloat());
}