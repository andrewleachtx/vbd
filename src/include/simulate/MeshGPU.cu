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
    freeGPUMem();
}

__host__ void MeshGPU::allocGPUMem(const Mesh& mesh, const SimConstants& h_consts) {
    size_t h_numVertices = mesh.cur_positions.size();

    // No need to free / reallocate if the size is the same; we should keep pointers to each MeshGPU instance
    if (h_numVertices > d_numVertices) {
        freeGPUMem();

        gpuErrchk(cudaMemcpyToSymbol(d_simConsts, &h_consts, sizeof(SimConstants)));

        gpuErrchk(cudaMalloc(&d_cur_positions, h_numVertices * sizeof(Eigen::Vector3f)));
        gpuErrchk(cudaMalloc(&d_prev_positions, h_numVertices * sizeof(Eigen::Vector3f)));
        gpuErrchk(cudaMalloc(&d_y, h_numVertices * sizeof(Eigen::Vector3f)));
        gpuErrchk(cudaMalloc(&d_cur_velocities, h_numVertices * sizeof(Eigen::Vector3f)));
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

        vector<int> h_v2tetWindow(d_numVertices * 2, -1);
        vector<int> h_v2tetIdx(total_ints, -1);
        size_t l(0), r(0);
        for (size_t i = 0; i < h_v2tetWindow.size(); i += 2) {
            r += mesh.vertex2tets[i].size();

            h_v2tetWindow[i] = l;
            h_v2tetWindow[i + 1] = r;

            // populate the actual values
            for (size_t j = 0; j < mesh.vertex2tets[i].size(); j++) {
                h_v2tetIdx[l + j] = mesh.vertex2tets[i][j];
            }
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
        d_tet_volumes = nullptr
    }

    d_numVertices = 0;
}

__device__ void MeshGPU::assembleVertexVForceAndHessian(const Eigen::Matrix<float, 9, 1>& dE_dF,
                                                         const Eigen::Matrix<float, 9, 9>& d2E_dF_dF,
                                                         float m1, float m2, float m3,
                                                         Eigen::Vector3f& force, Eigen::Matrix3f& h) {
    // f_ij = - d(Ej)/d(x_i)
    // H_ij = d2(E_j)/[d(x_i)d(x_i)]
    // f_i = SUM_j (f_ij)
    // H_i = SUM_j (H_ij)

    // Solve for delta_x_i = -H_i^-1 * f_i
    // Perform optional line search, but idk for now + paper said it was eh
    // x_i_new = x_i + delta_x_i
}

__device__ void MeshGPU::computeElasticEnergyGradients(float dt, size_t v_idx, size_t tet_idx,
                                                       Eigen::Vector3f& f_i, Eigen::Matrix3f& H_i) {
    // f_i = - d(G_i(x)) / d(x_i)) = - (m_i / h^2) * (x_i - y_i) - SUM( d(E_j(x)) / d(x_i) )
    // H_i (3x3) = d2(G_i(x)) / d(x_i)d(x_i) = (m_i / h^2) * I + SUM( d2(E_j(x)) / d(x_i)d(x_i) )
}

__global__ void MeshGPU::processVerticesKernel(Eigen::Vector3f* d_cur_positions, Eigen::Vector3f* d_prev_positions, Eigen::Vector3f* d_y,
                                               Eigen::Vector3f* d_cur_velocities, Eigen::Vector3f* d_forces, Eigen::Matrix3f* d_Hessian,
                                               const SimConstants consts, size_t numVertices, size_t start, size_t end) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    // FIXME: This wastes a lot of potential per launch; we could maybe just have these threads await some color update
    if (idx < start || idx >= end) {
        return;
    }

    // Embed into shared memory
    __shared__ Eigen::Vector3f s_cur_positions[THREADS_PER_BLOCK];
    __shared__ Eigen::Vector3f s_prev_positions[THREADS_PER_BLOCK];
    __shared__ Eigen::Vector3f s_y[THREADS_PER_BLOCK];
    __shared__ Eigen::Vector3f s_cur_velocities[THREADS_PER_BLOCK];
    __shared__ Eigen::Vector3f s_forces[THREADS_PER_BLOCK];
    __shared__ Eigen::Matrix3f s_Hessian[THREADS_PER_BLOCK];

    s_cur_positions[threadIdx.x] = d_cur_positions[idx];
    s_prev_positions[threadIdx.x] = d_prev_positions[idx];
    s_y[threadIdx.x] = d_y[idx];
    s_cur_velocities[threadIdx.x] = d_cur_velocities[idx];
    s_forces[threadIdx.x] = d_forces[idx];
    s_Hessian[threadIdx.x] = d_Hessians[idx];

    // Barrier to make sure shared memory is populated for this block
    __syncthreads();

    // Now do parallel for each j in F_i
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

    for (size_t c = 0; c < mesh.color_ranges.size(); c++) {
        size_t start(mesh.color_ranges[c][0]), end(mesh.color_ranges[c][1]);

        // Implicit ceiling FTW
        size_t blocks_per_grid = (end - start + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

        cout << "Launching kernel with " << blocks_per_grid << " blocks and " << THREADS_PER_BLOCK << " threads per block" << endl;
        /*
            parallel for vertex i in color c

            we need our kernel instance to only access i values from [start, end) by passing in these parameters

            it is inevitable that we are going to use an excess of threads perhaps, but we can try to bound that
        */

        processVerticesKernel<<<blocks_per_grid, THREADS_PER_BLOCK>>>(
            d_cur_positions, d_prev_positions, d_y, d_cur_velocities,
            d_forces, d_Hessians, d_simConsts, num_vertices, start, end
        );

        // We must barrier
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());
    }
}