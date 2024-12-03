#pragma once
#ifndef PHYSICS_SCENE_H
#define PHYSICS_SCENE_H

#include <string>
#include <vector>
class Mesh;

#define GLM_FORCE_RADIANS
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>

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

        // Our scene applies the same external forces, uses the same timestep, and uses the same number of max iterations
        
        std::vector<Mesh> meshes;
        glm::vec3 gravity;
        float dt;
        float iterations;
};

#endif // PHYSICS_SCENE_H
