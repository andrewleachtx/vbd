#pragma once
#ifndef PHYSICS_H
#define PHYSICS_H

#include <string>

class Physics {
    public:
        Physics(int scene_no, std::string state_output_dir);
        ~Physics() {}

        void init();
        void simulate();

    private:
        int scene_no;
        std::string state_output_dir;

};

#endif // PHYSICS_H
