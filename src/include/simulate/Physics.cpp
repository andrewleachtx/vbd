#include "Physics.h"

Physics::Physics(int scene_no, std::string state_output_dir) {
    this->scene_no = scene_no;
    this->state_output_dir = state_output_dir;
}

void Physics::init() {}
void Physics::simulate() {}