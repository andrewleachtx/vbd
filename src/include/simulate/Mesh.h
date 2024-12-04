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
        std::vector<int> old_indices;

        // The only velocity we need is "previous"
        std::vector<glm::vec3> velocities;


        // All tetrahedra are just int[4]
        std::vector<std::array<int, 4>> tetrahedra;

        // We can store the color of each vertex, and the index of any all colors per vector[color]
        std::vector<int> colors;
        std::vector<std::vector<int>> color_groups;

        // Now we have per mesh parameters        
        float mass, mu, lambda, damping, k_c, mu_c, eps_c;
        glm::vec3 position;
        glm::vec3 velocity;
        bool is_static;

        void initFromJson(const json& scene_data);
        void initFromVTK(const std::string& vtk_file);
};

#endif // MESH_H