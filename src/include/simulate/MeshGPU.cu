#include "MeshGPU.h"

#include "Mesh.h"
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

MeshGPU::MeshGPU() : d_cur_positions(nullptr), d_prev_positions(nullptr), d_y(nullptr),
                     d_cur_velocities(nullptr), d_forces(nullptr), d_Hessians(nullptr),
                     d_v2tetWindow(nullptr), d_v2tetIdx(nullptr), d_tetrahedra(nullptr),
                     d_Dm_inverses(nullptr), d_tet_volumes(nullptr), d_numVertices(0) {}

MeshGPU::~MeshGPU() {
    // freeGPUMem();
}

__host__ void MeshGPU::allocGPUMem(const Mesh& mesh, const SimConstants& h_consts) {
    size_t h_numVertices = mesh.cur_positions.size();

    // No need to free / reallocate if the size is the same; we should keep pointers to each MeshGPU instance
    if (h_numVertices > d_numVertices) {
        // freeGPUMem();

        gpuErrchk(cudaMemcpyToSymbol(d_simConsts, &h_consts, sizeof(SimConstants)));

        gpuErrchk(cudaMalloc(&d_cur_positions, h_numVertices * sizeof(Eigen::Vector3f)));
        gpuErrchk(cudaMalloc(&d_prev_positions, h_numVertices * sizeof(Eigen::Vector3f)));
        gpuErrchk(cudaMalloc(&d_y, h_numVertices * sizeof(Eigen::Vector3f)));
        gpuErrchk(cudaMalloc(&d_cur_velocities, h_numVertices * sizeof(Eigen::Vector3f)));
        gpuErrchk(cudaMalloc(&d_x_new, h_numVertices * sizeof(Eigen::Vector3f)));
        gpuErrchk(cudaMalloc(&d_forces, h_numVertices * sizeof(Eigen::Vector3f)));
        gpuErrchk(cudaMalloc(&d_Hessians, h_numVertices * sizeof(Eigen::Matrix3f)));

        /*
            Because this is sort of a constant call, we can also just copy the constant parts into global memory
            so that the copyToGPU doesn't unnecessarily do so - this is because the intention of copyToGPU is to
            do that
        */
        // Flatten vertex2tets and allocate 2x the space for indices - we have [i, i + 1] start, end 
        gpuErrchk(cudaMalloc(&d_v2tetWindow, h_numVertices * 2 * sizeof(int)));

        size_t total_ints = 0;
        for (const auto& vec : mesh.vertex2tets) {
            total_ints += vec.size();
        }
        
        gpuErrchk(cudaMalloc(&d_v2tetIdx, total_ints * sizeof(int)));
        gpuErrchk(cudaMalloc(&d_tetrahedra, 4 * mesh.tetrahedra.size() * sizeof(int)));
        gpuErrchk(cudaMalloc(&d_Dm_inverses, mesh.Dm_inverses.size() * sizeof(Eigen::Matrix3f)));
        gpuErrchk(cudaMalloc(&d_tet_volumes, mesh.tet_volumes.size() * sizeof(float)));

        vector<int> h_v2tetWindow(h_numVertices * 2, -1);
        vector<int> h_v2tetIdx(total_ints, -1);
        size_t l(0), r(0);
        for (size_t i = 0; i < mesh.vertex2tets.size(); i++) {
            r += mesh.vertex2tets[i].size();

            h_v2tetWindow[i] = l;
            h_v2tetWindow[i + 1] = r;

            // populate the actual values
            for (size_t j = 0; j < mesh.vertex2tets[i].size(); j++) {
                h_v2tetIdx[l + j] = mesh.vertex2tets[i][j];
            }

            l = r;
        }

        // Now we need to flatten the tetrahedra
        vector<int> h_tetrahedraFlat(4 * mesh.tetrahedra.size(), -1);
        for (size_t i = 0; i < mesh.tetrahedra.size(); i++) {
            for (size_t j = 0; j < 4; j++) {
                h_tetrahedraFlat[i * 4 + j] = mesh.tetrahedra[i][j];
            }
        }

        gpuErrchk(cudaMemcpy(d_v2tetWindow, h_v2tetWindow.data(), h_v2tetWindow.size() * sizeof(int), cudaMemcpyHostToDevice));
        gpuErrchk(cudaMemcpy(d_v2tetIdx, h_v2tetIdx.data(), h_v2tetIdx.size() * sizeof(int), cudaMemcpyHostToDevice));
        gpuErrchk(cudaMemcpy(d_tetrahedra, h_tetrahedraFlat.data(), h_tetrahedraFlat.size() * sizeof(int), cudaMemcpyHostToDevice));
        gpuErrchk(cudaMemcpy(d_Dm_inverses, mesh.Dm_inverses.data(), mesh.Dm_inverses.size() * sizeof(Eigen::Matrix3f), cudaMemcpyHostToDevice));
        gpuErrchk(cudaMemcpy(d_tet_volumes, mesh.tet_volumes.data(), mesh.tet_volumes.size() * sizeof(float), cudaMemcpyHostToDevice));

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

    // Initialize the forces & Hessians to zero
    vector<Eigen::Vector3f> f_zeroed(d_numVertices, Eigen::Vector3f::Zero());
    vector<Eigen::Matrix3f> H_zeroed(d_numVertices, Eigen::Matrix3f::Zero());
    gpuErrchk(cudaMemcpy(d_forces, f_zeroed.data(), d_numVertices * sizeof(Eigen::Vector3f), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_Hessians, H_zeroed.data(), d_numVertices * sizeof(Eigen::Matrix3f), cudaMemcpyHostToDevice));
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

    if (d_x_new) {
        gpuErrchk(cudaFree(d_x_new));
        d_x_new = nullptr;
    }

    if (d_forces) {
        gpuErrchk(cudaFree(d_forces));
        d_forces = nullptr;
    }

    if (d_Hessians) {
        gpuErrchk(cudaFree(d_Hessians));
        d_Hessians = nullptr;
    }

    if (d_v2tetWindow) {
        gpuErrchk(cudaFree(d_v2tetWindow));
        d_v2tetWindow = nullptr;
    }

    if (d_v2tetIdx) {
        gpuErrchk(cudaFree(d_v2tetIdx));
        d_v2tetIdx = nullptr;
    }

    if (d_tetrahedra) {
        gpuErrchk(cudaFree(d_tetrahedra));
        d_tetrahedra = nullptr;
    }

    if (d_Dm_inverses) {
        gpuErrchk(cudaFree(d_Dm_inverses));
        d_Dm_inverses = nullptr;
    }

    if (d_tet_volumes) {
        gpuErrchk(cudaFree(d_tet_volumes));
        d_tet_volumes = nullptr;
    }

    d_numVertices = 0;
}

__device__ __inline__ void assembleVertexVForceAndHessian(const Eigen::Matrix<float, 9, 1>& dE_dF,
                                                          const Eigen::Matrix<float, 9, 9>& d2E_dF_dF,
                                                          float m1, float m2, float m3,
                                                          Eigen::Vector3f& f_ij, Eigen::Matrix3f& H_ij) {
    float A1 = dE_dF(0);
    float A2 = dE_dF(1);
    float A3 = dE_dF(2);
    float A4 = dE_dF(3);
    float A5 = dE_dF(4);
    float A6 = dE_dF(5);
    float A7 = dE_dF(6);
    float A8 = dE_dF(7);
    float A9 = dE_dF(8);

    f_ij << A1 * m1 + A4 * m2 + A7 * m3,
             A2 * m1 + A5 * m2 + A8 * m3,
             A3 * m1 + A6 * m2 + A9 * m3;
    
    Eigen::Matrix<float, 3, 9> HL;

    HL.row(0) = d2E_dF_dF.row(0) * m1 + d2E_dF_dF.row(3) * m2 + d2E_dF_dF.row(6) * m3;
	HL.row(1) = d2E_dF_dF.row(1) * m1 + d2E_dF_dF.row(4) * m2 + d2E_dF_dF.row(7) * m3;
	HL.row(2) = d2E_dF_dF.row(2) * m1 + d2E_dF_dF.row(5) * m2 + d2E_dF_dF.row(8) * m3;

	H_ij.col(0) = HL.col(0) * m1 + HL.col(3) * m2 + HL.col(6) * m3;
	H_ij.col(1) = HL.col(1) * m1 + HL.col(4) * m2 + HL.col(7) * m3;
	H_ij.col(2) = HL.col(2) * m1 + HL.col(5) * m2 + HL.col(8) * m3;
}

__device__ void computeElasticEnergyGradients(size_t v_idx, int tet_idx, int* d_tetrahedra,
                                              Eigen::Vector3f* d_cur_positions,
                                              float* d_tet_volumes, Eigen::Matrix3f* d_Dm_inverses,
                                              Eigen::Vector3f& f_ij, Eigen::Matrix3f& H_ij) {
    // Because tet_idx is the start of four indices pointing we should grab those
    // TODO put simconsts vars in register variables even though cache probably does it
    int v0 = d_tetrahedra[tet_idx * 4];
    int v1 = d_tetrahedra[tet_idx * 4 + 1];
    int v2 = d_tetrahedra[tet_idx * 4 + 2];
    int v3 = d_tetrahedra[tet_idx * 4 + 3];
        
    Eigen::Vector3f x0 = d_cur_positions[v0];
    Eigen::Vector3f x1 = d_cur_positions[v1];
    Eigen::Vector3f x2 = d_cur_positions[v2];
    Eigen::Vector3f x3 = d_cur_positions[v3];

    float a = 1 + d_simConsts.mu / d_simConsts.lambda;
    float A = d_tet_volumes[tet_idx];

    Eigen::Matrix3f Dm_inv = d_Dm_inverses[tet_idx];
    Eigen::Matrix3f Ds;
    Ds.col(0) = x1 - x0;
    Ds.col(1) = x2 - x0;
    Ds.col(2) = x3 - x0;
    Eigen::Matrix3f F = Ds * Dm_inv;

    float detF = F.determinant();

    Eigen::Map<Eigen::Matrix<float, 9, 1>> dPhi_D_dF(F.data());

    float F1_1 = F(0, 0);
    float F2_1 = F(1, 0);
    float F3_1 = F(2, 0);
    float F1_2 = F(0, 1);
    float F2_2 = F(1, 1);
    float F3_2 = F(2, 1);
    float F1_3 = F(0, 2);
    float F2_3 = F(1, 2);
    float F3_3 = F(2, 2);

    Eigen::Matrix<float, 9, 1> ddetF_dF;
    ddetF_dF << F2_2 * F3_3 - F2_3 * F3_2,
                F1_3* F3_2 - F1_2 * F3_3,
                F1_2* F2_3 - F1_3 * F2_2,
                F2_3* F3_1 - F2_1 * F3_3,
                F1_1* F3_3 - F1_3 * F3_1,
                F1_3* F2_1 - F1_1 * F2_3,
                F2_1* F3_2 - F2_2 * F3_1,
                F1_2* F3_1 - F1_1 * F3_2,
                F1_1* F2_2 - F1_2 * F2_1;

    Eigen::Matrix<float, 9, 9> d2E_dF_dF = ddetF_dF * ddetF_dF.transpose();

    float k = detF - a;
    d2E_dF_dF(0, 4) += k * F3_3;
    d2E_dF_dF(4, 0) += k * F3_3;
    d2E_dF_dF(0, 5) += k * -F2_3;
    d2E_dF_dF(5, 0) += k * -F2_3;
    d2E_dF_dF(0, 7) += k * -F3_2;
    d2E_dF_dF(7, 0) += k * -F3_2;
    d2E_dF_dF(0, 8) += k * F2_2;
    d2E_dF_dF(8, 0) += k * F2_2;

    d2E_dF_dF(1, 3) += k * -F3_3;
    d2E_dF_dF(3, 1) += k * -F3_3;
    d2E_dF_dF(1, 5) += k * F1_3;
    d2E_dF_dF(5, 1) += k * F1_3;
    d2E_dF_dF(1, 6) += k * F3_2;
    d2E_dF_dF(6, 1) += k * F3_2;
    d2E_dF_dF(1, 8) += k * -F1_2;
    d2E_dF_dF(8, 1) += k * -F1_2;

    d2E_dF_dF(2, 3) += k * F2_3;
    d2E_dF_dF(3, 2) += k * F2_3;
    d2E_dF_dF(2, 4) += k * -F1_3;
    d2E_dF_dF(4, 2) += k * -F1_3;
    d2E_dF_dF(2, 6) += k * -F2_2;
    d2E_dF_dF(6, 2) += k * -F2_2;
    d2E_dF_dF(2, 7) += k * F1_2;
    d2E_dF_dF(7, 2) += k * F1_2;

    d2E_dF_dF(3, 7) += k * F3_1;
    d2E_dF_dF(7, 3) += k * F3_1;
    d2E_dF_dF(3, 8) += k * -F2_1;
    d2E_dF_dF(8, 3) += k * -F2_1;

    d2E_dF_dF(4, 6) += k * -F3_1;
    d2E_dF_dF(6, 4) += k * -F3_1;
    d2E_dF_dF(4, 8) += k * F1_1;
    d2E_dF_dF(8, 4) += k * F1_1;

    d2E_dF_dF(5, 6) += k * F2_1;
    d2E_dF_dF(6, 5) += k * F2_1;
    d2E_dF_dF(5, 7) += k * -F1_1;
    d2E_dF_dF(7, 5) += k * -F1_1;

    d2E_dF_dF *= d_simConsts.lambda;

    // Or d2E_dF_dF += mu * I_9x9

    d2E_dF_dF(0, 0) += d_simConsts.mu;
    d2E_dF_dF(1, 1) += d_simConsts.mu;
    d2E_dF_dF(2, 2) += d_simConsts.mu;
    d2E_dF_dF(3, 3) += d_simConsts.mu;
    d2E_dF_dF(4, 4) += d_simConsts.mu;
    d2E_dF_dF(5, 5) += d_simConsts.mu;
    d2E_dF_dF(6, 6) += d_simConsts.mu;
    d2E_dF_dF(7, 7) += d_simConsts.mu;
    d2E_dF_dF(8, 8) += d_simConsts.mu;

    Eigen::Matrix<float, 9, 1> dE_dF = A * (d_simConsts.mu * dPhi_D_dF + d_simConsts.lambda * (detF - a) * ddetF_dF);

    float DmInv1_1 = Dm_inv(0, 0);
    float DmInv2_1 = Dm_inv(1, 0);
    float DmInv3_1 = Dm_inv(2, 0);
    float DmInv1_2 = Dm_inv(0, 1);
    float DmInv2_2 = Dm_inv(1, 1);
    float DmInv3_2 = Dm_inv(2, 1);
    float DmInv1_3 = Dm_inv(0, 2);
    float DmInv2_3 = Dm_inv(1, 2);
    float DmInv3_3 = Dm_inv(2, 2);

    Eigen::Matrix<float, 9, 3> dF_dxi;
    Eigen::Vector3f dE_dxi;
    Eigen::Matrix3f d2E_dxi_dxi;
    float m1, m2, m3;

    if (v_idx == v0) {
        m1 = -DmInv1_1 - DmInv2_1 - DmInv3_1;
        m2 = -DmInv1_2 - DmInv2_2 - DmInv3_2;
        m3 = -DmInv1_3 - DmInv2_3 - DmInv3_3;
    }
    else if (v_idx == v1) {
        m1 = DmInv1_1;
        m2 = DmInv1_2;
        m3 = DmInv1_3;
    }
    else if (v_idx == v2) {
        m1 = DmInv2_1;
        m2 = DmInv2_2;
        m3 = DmInv2_3;
    }
    else if (v_idx == v3) {
        m1 = DmInv3_1;
        m2 = DmInv3_2;
        m3 = DmInv3_3;
    }

    assembleVertexVForceAndHessian(dE_dF, d2E_dF_dF, m1, m2, m3, dE_dxi, d2E_dxi_dxi);

    /*
        Damping term

        f_i -= (kd / h) * (d2E_dxi_dxi) * (x_i - xt_i)
        H_i += (kd / h) * (d2E_dxi_dxi)
    */

    // auto f_i_damping = (damping / dt) * (d2E_dxi_dxi * (cur_positions[v_idx] - y[v_idx]));
    // auto H_i_damping = (damping / dt) * d2E_dxi_dxi;
    // auto f_i_damping = Eigen::Vector3f::Zero();
    // auto H_i_damping = Eigen::Matrix3f::Zero();

    // print everything

    f_ij -= dE_dxi;
    H_ij += d2E_dxi_dxi;

    // if (v_idx == 0) {
    //     printvec3(f_ij);
    //     printmat3(H_ij);
    //     printvec3(dE_dxi);
    //     printmat3(d2E_dxi_dxi);
    //     printf("m1 = %f, m2 = %f, m3 = %f, k = %f\n", m1, m2, m3, k);
    //     printf("A = %f, a = %f\n", A, a);
    // }

}

__global__ void processVerticesKernel(Eigen::Vector3f* d_cur_positions, Eigen::Vector3f* d_prev_positions, Eigen::Vector3f* d_y,
                                      Eigen::Vector3f* d_cur_velocities, Eigen::Vector3f* d_x_new, Eigen::Vector3f* d_forces,
                                      Eigen::Matrix3f* d_Hessians, size_t numVertices,
                                      int* d_v2tetWindow, int* d_v2tetIdx,
                                      int* d_tetrahedra, Eigen::Matrix3f* d_Dm_inverses, float* d_tet_volumes,
                                      size_t start, size_t end) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    // FIXME: This wastes a lot of potential per launch; we could maybe just have these threads await some color update
    if (idx < start || idx >= end) {
        return;
    }

    // This is the index within the block (thread index)
    // size_t tIdx = threadIdx.x;

    // Embed into shared memory // FIXME: Until we sort vertices by color, it's actually pretty random what tetrahedra we access
    // __shared__ Eigen::Vector3f s_forces[THREADS_PER_BLOCK];
    // __shared__ Eigen::Matrix3f s_Hessians[THREADS_PER_BLOCK];
    // __shared__ Eigen::Vector3f s_cur_positions[THREADS_PER_BLOCK];
    // __shared__ Eigen::Vector3f s_y_positions[THREADS_PER_BLOCK];

    // s_forces[tIdx] = d_forces[idx];
    // s_Hessians[tIdx] = d_Hessians[idx];
    // s_cur_positions[tIdx] = d_cur_positions[idx];
    // s_y_positions[tIdx] = d_y[idx];

    // Barrier to make sure shared memory is populated for this block
    // __syncthreads();

    Eigen::Vector3f f_i = - (d_simConsts.mass * d_simConsts.inv_dtdt) * (d_cur_positions[idx] - d_y[idx]);
    Eigen::Matrix3f H_i = (d_simConsts.mass * d_simConsts.inv_dtdt) * Eigen::Matrix3f::Identity();

    // Too complex to add another layer of parallelization per tetrahedra - for now we will series
    for (size_t j = d_v2tetWindow[idx]; j < d_v2tetWindow[idx + 1]; j++) {
        // Keep in mind tet_idx is the start of 4 indices in d_tetrahedra[tet_idx : tet_idx + 4]
        size_t tet_idx = d_v2tetIdx[j];

        computeElasticEnergyGradients(idx, tet_idx, d_tetrahedra, d_cur_positions,
                                      d_tet_volumes, d_Dm_inverses, f_i, H_i);
    }

    if (H_i.determinant() > FLOAT_EPS) {
        const Eigen::Vector3f delta_xi = H_i.inverse() * f_i;

        d_x_new[idx] = d_cur_positions[idx] + delta_xi;
    }
    else {
        d_x_new[idx] = d_cur_positions[idx];
    }

    // Update position after barrier
    __syncthreads();
    d_cur_positions[idx] = d_x_new[idx];
}

/*
    Just as doVBDCPU calls its functions, we can call our device only functions from this
    launch.

    Following the paper's pseudocode below.
*/
__host__ void MeshGPU::doVBDGPU(float dt, Mesh& mesh) {
    size_t num_vertices = mesh.cur_positions.size();
    vector<Eigen::Vector3f> x_new(num_vertices, Eigen::Vector3f::Zero());
    
    for (size_t c = 0; c < mesh.color_ranges.size(); c++) {
        size_t start(mesh.color_ranges[c][0]), end(mesh.color_ranges[c][1]);

        // Implicit ceiling FTW
        size_t blocks_per_grid = (end - start + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

        /*
            parallel for vertex i in color c

            we need our kernel instance to only access i values from [start, end) by passing in these parameters

            it is inevitable that we are going to use an excess of threads perhaps, but we can try to bound that
        */

        processVerticesKernel<<<blocks_per_grid, THREADS_PER_BLOCK>>>(d_cur_positions, d_prev_positions, d_y, d_cur_velocities,
                                                                      d_x_new, d_forces, d_Hessians, num_vertices,
                                                                      d_v2tetWindow, d_v2tetIdx, d_tetrahedra, d_Dm_inverses,
                                                                      d_tet_volumes, start, end);

        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());
        // We barrier and then copy back at the end of the color in processVerticesKernel
    }

    // Copy back positions to CPU after its done done
    gpuErrchk(cudaMemcpy(mesh.cur_positions.data(), d_cur_positions, num_vertices * sizeof(Eigen::Vector3f), cudaMemcpyDeviceToHost));
}