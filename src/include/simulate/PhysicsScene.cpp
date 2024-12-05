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

    // Check to see the scene even exists
    if (scene_no >= scenes["scenes"].size()) {
        throw std::runtime_error("Scene number " + std::to_string(scene_no) + " does not exist in " + scene_file);
    }

    // Load the scene-specific properties ("properties")
    json& scene_data = scenes["scenes"][scene_no];
    json& scene_props = scene_data["properties"];

    // glm::vec3[1] == glm::vec3.y
    for (size_t i = 0; i < 3; i++) {
        gravity[i] = scene_props["gravity"][i];
    }
    dt = scene_props["solver"]["h"];
    iterations = scene_props["solver"]["iterations"];

    // Create and push back meshes - we should populate both mesh info and vertices / tetrahedra
    for (const auto& scene_mesh : scene_data["meshes"]) {
        Mesh mesh;
        mesh.initFromJson(scene_mesh);
        mesh.initFromVTK(resource_dir + "/models/vtk/" + string(scene_mesh["file"]));
        meshes.push_back(mesh);
    }

    cout << "Scene loaded successfully!" << endl;
}

/*
    CCD and DCD // TODO: Move to a collision.h + .cpp

    Based on penetration depth d, s.t.

    d = max(0, (dot(x_b - x_a, n_hat)))
    E_c(x) = 1/2 k_c d^2

    Let 
        - k_c be the collision stiffness parameter
        - x_a, x_b contact points on either side of the collision
        - n_hat contact normal

    Edge-edge collisions use CCD
        - x_a, x_b are the intersection points on either edge and the contact
          normal is the direction between them; n_hat = n / |n| where n = x_b - x_a
    Vertex-triangle collisions are detected either with CCD or DCD
        - x_a is the colliding vertex and x_b is the corresponding point on the collision point
          for CCD or the closest point for DCD on the triangle's face. n_hat is the surface normal at x_b.
*/
struct Collision {
    glm::vec3 x_a, x_b, n_hat;
    float d;
};

void vertexTriangleCollision(const Mesh& mesh_a, const Mesh& mesh_b, Collision& collision) {
    
}

void PhysicsScene::discreteCollisionDetection() {
    // Check between all meshes if there is a vertex-triangle collision
    for (size_t i = 0; i < meshes.size(); i++) {
        for (size_t j = i + 1; j < meshes.size(); j++) {
        }
    }

}
void PhysicsScene::continuousCollisionDetection() {}

/*
    Input: 
        x_t pos of previous step
        v_t vel of prev step
        a_ext external acceleration (gravity for now)
    
    Output: this step's position & velocity x_t+1, v_t+1 (not really an output cause void)
*/
void PhysicsScene::stepCPU() {
    // TODO: Base this on the paper's pseudocode

    int max_substeps(10);
    for (int substep = 0; substep < max_substeps; substep++) {
        // Do discrete collision detection
        // discreteCollisionDetection();

        /*
            If we are "ready to solve" => "those are for CCD, to apply CCD the vertex must be inversion free in both prev position and current position"
            then for each tet 1) evaluate external forces 2) apply initial step / guess

            for 2) see applyInitialStep(), storet his in "x"
                - This is the adaptive initialization calculations of a~ to be stored in x
        */
        for (Mesh& mesh : meshes) {
            mesh.initialGuessAdaptive(dt, gravity);
        }
        cout << "Done with initial guess!" << endl;

        // for iter in max iterations
            // if n mod n_collision // TODO: add n_col to physics_scene then perform CCD with x

            // for each vertex color / group c
                // parallel for each vertex i in color c
                    // parallel for each j in F_i => we can avoid the SUM( ... ) below by doing it and joining later 
                        /*
                            This part is more involved with Hessian, see VBDSolveParallelGroup_allInOne_kernel_V2

                            f_i = - d(G_i(x)) / d(x_i)) = - (m_i / h^2) * (x_i - y_i) - SUM( d(E_j(x)) / d(x_i) )
                            H_i (3x3) = d2(G_i(x)) / d(x_i)d(x_i) = (m_i / h^2) * I + SUM( d2(E_j(x)) / d(x_i)d(x_i) )

                        */
                        // f_ij = - d(Ej)/d(x_i)
                        // H_ij = d2(E_j)/[d(x_i)d(x_i)]
                    
                    // join reduction sums
                    // f_i = SUM_j (f_ij)
                    // H_i = SUM_j (H_ij)

                    // Solve for delta_x_i = -H_i^-1 * f_i
                    // Perform optional line search, for now idk
                    // x_i_new = x_i + delta_x_i

                // Because x_i_new is an external buffer so other colors don't trip:
                // parallel for each vertex i in color c do
                    // x_i = x_i_new
            

            // Optionally, you can use the "accelerated" iteration
            // parallel for each vertex i do (update x_i using Eqn 18)
        for (int iter = 0; iter < iterations; iter++) {
            for (Mesh& mesh : meshes) {
                mesh.doVBDCPU(dt);
            }
        }
        
        // v_t = (x - x_t) / h
        for (Mesh& mesh : meshes) {
            mesh.updateVelocities(dt);
        }
    }
}

/*
    For now this will run on the CPU
*/
void PhysicsScene::simulate() {
    // TODO: Move max_frames to attribute
    int cur_frame(0), max_frames(10);

    while (++cur_frame < max_frames) {
        stepCPU();
        // stepGPU();
    }

}

