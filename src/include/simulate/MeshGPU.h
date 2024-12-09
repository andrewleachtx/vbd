#pragma once
#ifndef MESH_GPU_H
#define MESH_GPU_H

#include <string>
#include <vector>
#include <Eigen/Dense>

#include <cuda_runtime.h>

struct SimConstants {
    float dt;
    float inv_dt;
    float inv_dtdt;
    float mass;
    float mu;
    float lambda;
    float damping;
    float k_c;
    float mu_c;
    float eps_c;
};

class Mesh;

/*
    From the original code if the GPU flag is true we can pass in the Mesh and perform copies for the GPU

    When the actual parallelized doVBDGPU is called, we will need 
        - vector<float> tet_volumes
        - vector<Eigen::Matrix3f> Dm_inverses
        - vector<array<int, 4>> tetrahedra
        - vector<vector<int>> vertex2tets

        - vector<Eigen::Vector3f> cur_positions
        - vector<Eigen::Vector3f> prev_positions
        - vector<Eigen::Vector3f> y
        - vector<Eigen::Vector3f> cur_velocities

        - Eigen::Vector3f* d_forces
        - Eigen::Matrix3f* d_Hessian

    Oh I just realized using d for device and also d for derivative is gonna get weird...
*/
class MeshGPU {
    public:
        MeshGPU();
        ~MeshGPU();

        __host__ void allocGPUMem(size_t h_numVertices, const SimConstants &h_consts);

        __host__ void allocGPUMem(size_t h_numVertices);
        __host__ void copyToGPU(const Mesh& mesh);
        __host__ void freeGPUMem();

        __host__ void doVBDGPU(float dt, Mesh& mesh);

        Eigen::Vector3f* d_cur_positions;
        Eigen::Vector3f* d_prev_positions;
        Eigen::Vector3f* d_y;
        Eigen::Vector3f* d_cur_velocities;

        Eigen::Vector3f* d_forces;
        Eigen::Matrix3f* d_Hessian;

        size_t d_numVertices;
};

#endif // MESH_GPU_H