#pragma once
#ifndef MESH_H
#define MESH_H

#include <string>
#include <vector>
#include <nlohmann/json.hpp>
using json = nlohmann::json;

class Mesh;

#define GLM_FORCE_RADIANS
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>


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
        std::vector<glm::vec3> init_positions;
        std::vector<glm::vec3> prev_positions;
        std::vector<glm::vec3> cur_positions;

        // Inertia + acceleration for the "ground truth" y
        std::vector<glm::vec3> y;

        std::vector<glm::vec3> prev_velocities;
        std::vector<glm::vec3> cur_velocities;

        // All tetrahedra are just int[4], and we can store all tetrahedra that correspond to one vertex idx
        std::vector<std::array<int, 4>> tetrahedra;
        std::vector<std::vector<int>> vertex2tets;

        // We can store the color of each vertex, and after sorting by color, the range of vertices per group
        std::vector<int> colors;
        std::vector<std::array<size_t, 2>> color_ranges;

        // Now we have per mesh parameters        
        float mass, mu, lambda, damping, k_c, mu_c, eps_c;
        glm::vec3 position;
        glm::vec3 velocity;
        bool is_static;

        void initialGuessAdaptive(float dt, const glm::vec3& a);
        void updateVelocities(float dt);

        void initFromJson(const json& scene_data);
        void initFromVTK(const std::string& vtk_file);
};

#endif // MESH_H