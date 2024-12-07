#pragma once
#ifndef MESH_H
#define MESH_H

#include <string>
#include <vector>
#include <nlohmann/json.hpp>
#include <Eigen/Dense>

using json = nlohmann::json;

class Mesh;

/*
    One mesh consists of multiple vertices, and multiple tetrahedra which are loaded externally in addition
    to the vertices themself.

    Basic SOA, but could even move towards a vector<float> pos_x, vector<float> pos_y, ... depending on what
    the CUDA accesses look like 

    Material properties for the mesh are read in from the json only, hence the lack of a constructor
*/
class Mesh {
    public:
        // For any vertex we should store the initial position, and use a buffer for previous and current for updating
        std::vector<Eigen::Vector3f> init_positions;
        std::vector<Eigen::Vector3f> prev_positions;
        std::vector<Eigen::Vector3f> cur_positions;

        // Inertia + acceleration for the "ground truth" y
        std::vector<Eigen::Vector3f> y;

        std::vector<Eigen::Vector3f> prev_velocities;
        std::vector<Eigen::Vector3f> cur_velocities;

        // All tetrahedra are just int[4], and we can store all tetrahedra that correspond to one vertex idx
        std::vector<std::array<int, 4>> tetrahedra;
        std::vector<std::vector<int>> vertex2tets;
        std::vector<Eigen::Matrix3f> Dm_inverses;
        std::vector<float> tet_volumes;

        // We can store the color of each vertex, and after sorting by color, the range of vertices per group
        std::vector<int> colors;
        std::vector<std::array<size_t, 2>> color_ranges;

        // Now we have per mesh parameters        
        float mass, mu, lambda, damping, k_c, mu_c, eps_c;
        Eigen::Vector3f position;
        Eigen::Vector3f velocity;
        bool is_static;

        enum initGuessEnum {
            INERTIA_ACCEL,
            ADAPTIVE
        };

        initGuessEnum initGuessType;

        void assembleVertexVForceAndHessian(const Eigen::Matrix<float, 9, 1>& dE_dF,
                                            const Eigen::Matrix<float, 9, 9>& d2E_dF_dF,
                                            float m1, float m2, float m3,
                                            Eigen::Vector3f& force, Eigen::Matrix3f& h);
        void computeElasticEnergyGradients(float dt, size_t v_idx, size_t tet_idx,
                                         Eigen::Vector3f& force, Eigen::Matrix3f& hessian);

        void computeElasticEnergyGradients(int v_idx, Eigen::Vector3f& f_i_elastic, Eigen::Matrix3f& H_i_elastic);
        void doVBDCPU(float dt);

        void initialGuess(float dt, const Eigen::Vector3f& a);
        void updateVelocities(float dt);

        void writeToVTK(const std::string& output_dir, bool raw=false);
        void initFromJson(const json& scene_data);
        void initFromVTK(const std::string& vtk_file);
};

#endif // MESH_H