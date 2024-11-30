#pragma once
#ifndef PHYSICS_SCENE_H
#define PHYSICS_SCENE_H

#include <string>
#include <vector>

#define GLM_FORCE_RADIANS
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>


// TODO: Move to proper mesh.h/cpp
/*
    For now we just need to match our per-mesh data specified in the file.

    To avoid mesh duplication, we can store a meshID that corresponds to a unique int : Mesh
*/
struct MeshRef {
    int underyling_meshId;
    std::string mesh_name;

};

/*
    This is the heart of the program.

    We can arbit our physics parameters according to a "scene" which is dictated by the
    specific number in scene_no - by default it reads from scenes/scenes.json

*/
class PhysicsScene {
    public:
        PhysicsScene(const std::string& resource_dir, int scene_no, const std::string& state_output_dir);
        ~PhysicsScene() {}

        void init();
        void simulate();

    private:
        std::string resource_dir;
        int scene_no;
        std::string state_output_dir;
        std::vector<int> meshes;

        // TODO: Each object specified in the json has its own data, we need an efficient way to store
        std::vector<glm::vec3> positions;
        std::vector<glm::vec3> velocities;
        std::vector<glm::vec3> accelerations;
};

#endif // PHYSICS_SCENE_H
