#pragma once
#ifndef PHYSICS_SCENE_H
#define PHYSICS_SCENE_H

#include <string>
#include <vector>
#include "Mesh.h"

#include <Eigen/Dense>

/*
    This is the heart of the program.

    We can arbit our physics parameters according to a "scene" which is dictated by the
    specific number in scene_no - by default it reads from scenes/scenes.json

*/
class PhysicsScene {
    public:
        PhysicsScene(const std::string& resource_dir, int scene_no, const std::string& state_output_dir,
                     bool is_usingGPU);
        ~PhysicsScene() {}

        void init();

        void discreteCollisionDetection();
        void continuousCollisionDetection();
        void runStepsGPU();
        void runStepsCPU();
        void simulate();

    private:
        std::string resource_dir;
        int scene_no;
        std::string state_output_dir;

        // Our scene applies the same external forces, uses the same timestep, and uses the same number of max iterations
        std::vector<Mesh> meshes;
        Eigen::Vector3f gravity;
        float dt;
        float iterations;

        size_t frame;
        bool is_usingGPU;
};

#endif // PHYSICS_SCENE_H
