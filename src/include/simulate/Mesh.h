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

    Highly structure of arrays with hopes of CUDA parallelization.

    Material properties for the mesh are read in from the json only, hence the lack of a constructor
*/
class Mesh {
    public:
        std::vector<float> initpos_x, initpos_y, initpos_z;
        std::vector<float> pos_x, pos_y, pos_z;
        std::vector<int> tet_v1, tet_v2, tet_v3, tet_v4;
        std::vector<int> colors;

        float mass, mu, lambda, damping, k_c, mu_c, eps_c;
        glm::vec3 position;
        glm::vec3 velocity;

        void initFromJson(const json& scene_data);
        void initFromVTK(const std::string& vtk_file);
};

#endif // MESH_H