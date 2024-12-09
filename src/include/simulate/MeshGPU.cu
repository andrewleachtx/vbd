#include "MeshGPU.h"

#include "../utils/utils.h"

#include <iostream>
#include <string>
#include <vector>
#include <Eigen/Dense>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

using std::cout, std::cerr, std::endl, std::string, std::vector, std::array;

/*
    Before we launch each kernel we know the Mesh information is constant for this doVBDGPU
    call; we can copy this information to the device in __constant__ space
*/
__constant__ SimConstants d_simConsts;

MeshGPU::MeshGPU() : d_cur_positions(nullptr), d_prev_positions(nullptr), d_y(nullptr), d_cur_velocities(nullptr), 
                     d_forces(nullptr), d_Hessian(nullptr), d_numVertices(0) {}
    
MeshGPU::~MeshGPU() {
    freeGPUMem();
}

__host__ void MeshGPU::allocGPUMem(size_t h_numVertices, const SimConstants& h_consts) {
    // No need to free / reallocate if the size is the same; we should keep pointers to each MeshGPU instance
    if (h_numVertices > d_numVertices) {
        freeGPUMem();

        gpuErrchk(cudaMemcpyToSymbol(d_simConsts, &h_consts, sizeof(SimConstants)));

        gpuErrchk(cudaMalloc(&d_cur_positions, h_numVertices * sizeof(Eigen::Vector3f)));
        gpuErrchk(cudaMalloc(&d_prev_positions, h_numVertices * sizeof(Eigen::Vector3f)));
        gpuErrchk(cudaMalloc(&d_y, h_numVertices * sizeof(Eigen::Vector3f)));
        gpuErrchk(cudaMalloc(&d_cur_velocities, h_numVertices * sizeof(Eigen::Vector3f)));
        gpuErrchk(cudaMalloc(&d_forces, h_numVertices * sizeof(Eigen::Vector3f)));
        gpuErrchk(cudaMalloc(&d_Hessian, h_numVertices * sizeof(Eigen::Matrix3f)));

        d_numVertices = h_numVertices;
    }
}

void MeshGPU::copyToGPU(const Mesh& mesh) {
    // Update only buffers that need it
    const vector<Eigen::Vector3f>& cur_pos = mesh.cur_positions;
    const vector<Eigen::Vector3f>& prev_pos = mesh.prev_positions;
    const vector<Eigen::Vector3f>& y = mesh.y;
    const vector<Eigen::Vector3f>& cur_vel = mesh.cur_velocities;

    gpuErrchk(cudaMemcpy(d_cur_positions, cur_pos.data(), d_numVertices * sizeof(Eigen::Vector3f), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_prev_positions, prev_pos.data(), d_numVertices * sizeof(Eigen::Vector3f), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_y, y.data(), d_numVertices * sizeof(Eigen::Vector3f), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_cur_velocities, cur_vel.data(), d_numVertices * sizeof(Eigen::Vector3f), cudaMemcpyHostToDevice));
}

void MeshGPU::freeGPUMem() {
    if (d_cur_positions) {
        gpuErrchk(cudaFree(d_cur_positions));
        d_cur_positions = nullptr;
    }

    if (d_prev_positions) {
        gpuErrchk(cudaFree(d_prev_positions));
        d_prev_positions = nullptr;
    }

    if (d_y) {
        gpuErrchk(cudaFree(d_y));
        d_y = nullptr;
    }

    if (d_cur_velocities) {
        gpuErrchk(cudaFree(d_cur_velocities));
        d_cur_velocities = nullptr;
    }

    if (d_forces) {
        gpuErrchk(cudaFree(d_forces));
        d_forces = nullptr;
    }

    if (d_Hessian) {
        gpuErrchk(cudaFree(d_Hessian));
        d_Hessian = nullptr;
    }

    d_numVertices = 0;
}

/*
    Just as doVBDCPU calls its functions, we can call our device only functions from this
    launch.

    Following the paper's pseudocode below.
*/
__host__ void MeshGPU::doVBDGPU(float dt, Mesh& mesh) {
    // for each vertex color / group c
        // parallel for each vertex i in color c
            // parallel for each j in F_i => we can avoid the SUM( ... ) below by doing it and joining later 
                /*
                    This part is more involved with Hessian, see VBDSolveParallelGroup_allInOne_kernel_V2

                    f_i = - d(G_i(x)) / d(x_i)) = - (m_i / h^2) * (x_i - y_i) - SUM( d(E_j(x)) / d(x_i) )
                    H_i (3x3) = d2(G_i(x)) / d(x_i)d(x_i) = (m_i / h^2) * I + SUM( d2(E_j(x)) / d(x_i)d(x_i) )

                */
                // f_ij = - d(Ej)/d(x_i)
                // H_ij = d2(E_j)/[d(x_i)d(x_i)]
            
            // join reduction sums
            // f_i = SUM_j (f_ij)
            // H_i = SUM_j (H_ij)

            // Solve for delta_x_i = -H_i^-1 * f_i
            // Perform optional line search, but idk for now + paper said it was eh
            // x_i_new = x_i + delta_x_i

        // Because x_i_new is an external buffer so other colors don't trip:
        // parallel for each vertex i in color c do
            // x_i = x_i_new

    float inv_dt = 1.0f / dt;
    float inv_dtdt = inv_dt * inv_dt;
    size_t num_vertices = mesh.cur_positions.size();
    vector<Eigen::Vector3f> x_new(num_vertices, Eigen::Vector3f::Zero());
    float m_i = mesh.mass;

    /*
        1) Allocate memory on device
        2) Copy host->device
        3) Launch kernels
        4) Copy device->host
    */
    // (1)

    SimConstants h_simconsts = { dt, inv_dt, inv_dtdt, m_i, mesh.mu, mesh.lambda, mesh.damping, mesh.k_c, mesh.mu_c, mesh.eps_c };

    allocGPUMem(num_vertices, h_simconsts);

    // (2)
    copyToGPU(mesh);

    // (3)

    

    
};