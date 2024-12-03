#pragma once
#ifndef UTILS_H
#define UTILS_H

#include "../../include.h"

#include <cuda_runtime.h>
#include <glm/glm.hpp>
#include <string>
using std::string;

float randFloat();
glm::vec3 randXYZ();

// https://stackoverflow.com/questions/14038589/what-is-the-canonical-way-to-check-for-errors-using-the-cuda-runtime-api/14038590#14038590
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
__inline __host__ void gpuAssert(cudaError_t code, const char *file, int line, 
                                 bool abort=true) {
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"[ERROR] gpuAssert: %s %s %d\n", cudaGetErrorString(code),
          file, line);
      if (abort) exit(code);
   }
}

#define printvec3(var) pv3(#var, var)
__inline __host__ __device__ void pv3(const char* varname, glm::vec3 vec) {
    printf("%s: %f, %f, %f\n", varname, vec.x, vec.y, vec.z);
}

#define printmat3(var) pm3(#var, var)
__inline __host__ __device__ void pm3(const char* varname, glm::mat3 mat) {
    printf("%s:\n", varname);
    printf("%f, %f, %f\n", mat[0][0], mat[0][1], mat[0][2]);
    printf("%f, %f, %f\n", mat[1][0], mat[1][1], mat[1][2]);
    printf("%f, %f, %f\n", mat[2][0], mat[2][1], mat[2][2]);
}

#define printmat4(var) pm4(#var, var)
__inline __host__ __device__ void pm4(const char* varname, glm::mat4 mat) {
    printf("%s:\n", varname);
    printf("%f, %f, %f\n", mat[0][0], mat[0][1], mat[0][2]);
    printf("%f, %f, %f\n", mat[1][0], mat[1][1], mat[1][2]);
    printf("%f, %f, %f\n", mat[2][0], mat[2][1], mat[2][2]);
    printf("%f, %f, %f\n", mat[3][0], mat[3][1], mat[3][2]);
}


#endif // UTILS_H