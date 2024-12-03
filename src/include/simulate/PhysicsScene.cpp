#include "PhysicsScene.h"
#include <cassert>
#include <iostream>
#include <fstream>
#include <nlohmann/json.hpp>
#include <omp.h>

#include "Mesh.h"

using std::cout, std::cerr, std::endl, std::string, std::vector, std::array, std::ifstream;
using json = nlohmann::json;

PhysicsScene::PhysicsScene(const string& resource_dir, int scene_no, const string& state_output_dir) {
    this->resource_dir = resource_dir;
    this->scene_no = scene_no;
    this->state_output_dir = state_output_dir;
}

void PhysicsScene::init() {
    assert(scene_no >= 0 && scene_no < 3);
    assert(resource_dir != "");
    assert(state_output_dir != "");

    // Load scene parameters
    string scene_file = resource_dir + "/scenes/scenes.json";
    std::ifstream fin(scene_file);
    if (!fin.is_open() || !fin.good()) {
        throw std::runtime_error("Error opening scene file: " + scene_file);
    }

    json scenes;
    fin >> scenes;
    fin.close();

    // Load the scene-specific properties ("properties")
    json& scene_data = scenes["scenes"][scene_no];
    json& scene_props = scene_data["properties"];

    // glm::vec3[1] == glm::vec3.y
    for (size_t i = 0; i < 3; i++) {
        gravity[i] = scene_props["gravity"][i];
    }
    dt = scene_props["solver"];
    iterations = scene_props["iterations"];

    // Create and push back meshes - we should populate both mesh info and vertices / tetrahedra
    for (const auto& scene_mesh : scene_data["meshes"]) {
        Mesh mesh;
        mesh.initFromJson(scene_mesh);
        mesh.initFromVTK(resource_dir + "/models/vtk/" + string(scene_mesh["file"]));
        meshes.push_back(mesh);
    }
}

void PhysicsScene::simulate() {
    
}

