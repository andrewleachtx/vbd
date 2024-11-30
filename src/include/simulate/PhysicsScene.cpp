#include "PhysicsScene.h"
#include <cassert>
#include <iostream>
#include <fstream>
#include <nlohmann/json.hpp>

using std::cout, std::cerr, std::endl, std::string, std::ifstream;
using json = nlohmann::json;

PhysicsScene::PhysicsScene(const std::string& resource_dir, int scene_no, const std::string& state_output_dir) {
    this->resource_dir = resource_dir;
    this->scene_no = scene_no;
    this->state_output_dir = state_output_dir;
}

void PhysicsScene::init() {
    assert(scene_no >= 0 && scene_no < 3);
    assert(resource_dir != "");
    assert(state_output_dir != "");

    // Load scene parameters
    std::string scene_file = resource_dir + "/scenes/scenes.json";
    std::ifstream fin(scene_file);
    if (!fin.is_open() || !fin.good()) {
        throw std::runtime_error("Error opening scene file: " + scene_file);
    }

    json scenes;
    fin >> scenes;
    fin.close();

    

}

void PhysicsScene::simulate() {
    
}

