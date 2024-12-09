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
        - Eigen::Matrix3f* d_Hessians

    Oh I just realized using d for device and also d for derivative is gonna get weird...
*/
class MeshGPU {
    public:
        MeshGPU();
        ~MeshGPU();

        __host__ void allocGPUMem(const Mesh& mesh, const SimConstants &h_consts);
        __host__ void copyToGPU(const Mesh& mesh);
        __host__ void freeGPUMem();

        // __device__ void assembleVertexVForceAndHessian(const Eigen::Matrix<float, 9, 1>& dE_dF,
        //                                     const Eigen::Matrix<float, 9, 9>& d2E_dF_dF,
        //                                     float m1, float m2, float m3,
        //                                     Eigen::Vector3f& force, Eigen::Matrix3f& h);
        
        __global__ void processVerticesKernel(Eigen::Vector3f* d_cur_positions, Eigen::Vector3f* d_prev_positions, Eigen::Vector3f* d_y,
                                              Eigen::Vector3f* d_cur_velocities, Eigen::Vector3f* d_forces, Eigen::Matrix3f* d_Hessians,
                                              const SimConstants consts, size_t numVertices,
                                              int* d_v2tetWindow, int* d_v2tetIdx,
                                              int* d_tetrahedra, Eigen::Matrix3f* d_Dm_inverses, float* d_tet_volumes,
                                              size_t start, size_t end);
        __host__ void doVBDGPU(float dt, Mesh& mesh);

        Eigen::Vector3f* d_cur_positions;
        Eigen::Vector3f* d_prev_positions;
        Eigen::Vector3f* d_y;
        Eigen::Vector3f* d_cur_velocities;

        Eigen::Vector3f* d_forces;
        Eigen::Matrix3f* d_Hessians;

        /*
            To access all associated tetrahedrons given a vertex with vector<vector<int>> vertex2tets[i] we can
            store a flattened array d_v2tetWindow[] such that start = d_v2tetWindow[i], end = d_v2tetWindow[i + 1]

            Those values start, end correspond to the flattened region of a d_vertex d_v2tetIdx[start:end] which
            contains the indices of the tetrahedra that contain vertex i

            Our tetrahedra are std::array<int, 4> with 4 indices of each vertex, so we can store a flattened int*
            for tetrahedra, with strides of 4 for each index.
        */
        int* d_v2tetWindow;
        int* d_v2tetIdx;
        int* d_tetrahedra;

        Eigen::Matrix3f* d_Dm_inverses;
        float* d_tet_volumes;
        
        size_t d_numVertices;
};

#endif // MESH_GPU_H