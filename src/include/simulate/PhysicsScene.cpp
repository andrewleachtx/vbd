#include "PhysicsScene.h"
#include <cassert>
#include <iostream>
#include <fstream>
#include <nlohmann/json.hpp>
#include <omp.h>
#include <chrono>

#include "../utils/utils.h"
#include "Mesh.h"

// CUDA
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "MeshGPU.h"

using std::cout, std::cerr, std::endl, std::string, std::vector, std::array, std::ifstream;
using json = nlohmann::json;

PhysicsScene::PhysicsScene(const string& resource_dir, int scene_no, const string& state_output_dir,
                           bool is_usingGPU) {
    this->resource_dir = resource_dir;
    this->scene_no = scene_no;
    this->state_output_dir = state_output_dir;
    this->is_usingGPU = is_usingGPU;
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

    gravity = Eigen::Vector3f(
        scene_props["gravity"][0].get<float>(),
        scene_props["gravity"][1].get<float>(),
        scene_props["gravity"][2].get<float>()
    );
    dt = scene_props["solver"]["h"];
    iterations = scene_props["solver"]["iterations"];

    // Create and push back meshes - we should populate both mesh info and vertices / tetrahedra
    for (const auto& scene_mesh : scene_data["meshes"]) {
        Mesh mesh;
        mesh.initFromJson(scene_mesh);
        mesh.initFromVTK(resource_dir + "/models/vtk/" + string(scene_mesh["file"]));
        meshes.push_back(mesh);
    }
    
    // For output purposes, we start at frame 0
    frame = 0;
    cout << "Scene loaded successfully! Parameters:" << endl;
    printvec3(gravity);
    cout << "\tTimestep: " << dt << endl;
    cout << "\tIterations: " << iterations << endl;
    cout << "\tMeshes: " << meshes.size() << endl;
    cout << "\tOutput directory: " << state_output_dir << endl;
}

/*
    Make one call at the start so we don't have to do it each step
*/
void PhysicsScene::initGPUMeshes() {
    meshesGPU.clear();
    meshesGPU.reserve(meshes.size());

    for (const Mesh& mesh : meshes) {
        MeshGPU gpu_mesh;
        meshesGPU.push_back(gpu_mesh);

        float inv_dt = 1.0f / dt;
        float inv_dtdt = inv_dt * inv_dt;
        float m_i = mesh.mass;

        SimConstants h_simconsts = { dt, inv_dt, inv_dtdt, m_i, mesh.mu, mesh.lambda, mesh.damping, mesh.k_c, mesh.mu_c, mesh.eps_c };
        meshesGPU[meshesGPU.size() - 1].allocGPUMem(mesh, h_simconsts);
    } 
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
void PhysicsScene::discreteCollisionDetection() {
    // Check between all meshes if there is a vertex-triangle collision
    for (size_t i = 0; i < meshes.size(); i++) {
        for (size_t j = i + 1; j < meshes.size(); j++) {
        }
    }

}
void PhysicsScene::continuousCollisionDetection() {}

/*
    For simplicity I will be using a hybrid approach that requires memcpy'ing around information
    and using kernels and transfers where the copy is cheaper than the computation.

    TODO: Do a proper implementation, storing all vertex data in one GPU malloc call. May be
    TODO: possible to also make one massive global memory alloc call at the start and index
    TODO: according to the mesh we are using. This would help a lot with the cudaMalloc call
    TODO: per mesh 
*/
void PhysicsScene::runStepsGPU() {
    int max_substeps(1);
    float sim_dt = dt / max_substeps;

    if (meshesGPU.size() == 0) {
        initGPUMeshes();
    }

    assert(meshesGPU.size() == meshes.size());

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
            mesh.initialGuess(sim_dt, gravity);
        }
        printvec3(meshes[0].cur_positions[0]);

        for (int iter = 0; iter < iterations; iter++) {
            /*
                Now we assume each gpu mesh has been alloc'd before the step, now each timestep
                all we need to do is copy over, launch kernel, and copy back (the copy is in
                VBDGPU as I pass mesh by reference
            */

            for (size_t i = 0; i < meshesGPU.size(); i++) {
                meshesGPU[i].copyToGPU(meshes[i]);
                meshesGPU[i].doVBDGPU(dt, meshes[i]);
            }
        }

        for (Mesh& mesh : meshes) {
            mesh.updateVelocities(sim_dt);
        }
    }
}

/*
    Input: 
        x_t pos of previous step
        v_t vel of prev step
        a_ext external acceleration (gravity for now)
    
    Output: this step's position & velocity x_t+1, v_t+1 (not really an output cause void)
*/
void PhysicsScene::runStepsCPU() {
    int max_substeps(5);
    float sim_dt = dt / max_substeps;
    // scenes.json "h": 0.00333333333333

    float rho = 0.95f;

    for (int substep = 0; substep < max_substeps; substep++) {
        for (Mesh& mesh : meshes) {
            mesh.initialGuess(sim_dt, gravity);
        }
        printvec3(meshes[0].cur_positions[0]);

        // Use omega for accelerated - make sure position update uncomments in Mesh.cpp
        float omega_cur(1.0f), omega_prev;
        for (int iter = 0; iter < iterations; iter++) {
            if (iter == 1) {
                omega_cur = (2.0f / (2.0f - (rho * rho)));
                omega_prev = 1.0f;
            }
            else {
                float tmp = omega_cur;
                omega_cur = (4.0f / (4.0f - (rho * rho) * omega_prev));
                omega_prev = tmp;
            }

            for (Mesh& mesh : meshes) {
                mesh.omega = omega_cur;
                mesh.doVBDCPU(sim_dt);
            }
        }
        
        for (Mesh& mesh : meshes) {
            mesh.updateVelocities(sim_dt);
        }
    }
}

/*
    For now this will run on the CPU
*/
void PhysicsScene::simulate() {
    // TODO: Move max_frames to attribute
    int max_frames(600);

    // Perturb
    float perturb = 0.35f;
    // Perturb tetrahedra
    for (size_t i = 0; i < meshes[0].prev_positions.size(); i++) {
        if (i % (meshes[0].prev_positions.size() / 10) == 0) {
            meshes[0].prev_positions[i] += perturb * randXYZ(true);
        }
    }

    // Flatten
    // for (size_t i = 0; i < meshes[0].prev_positions.size(); i++) {
    //     meshes[0].prev_positions[i].y() = 0.0f;
    // }

    int waiting_frames = 89;

    // If we want 1.5 seconds of frozen at first with 60 fps let the first 90 frames be subtle
    while (++frame < waiting_frames) {
        string filename = state_output_dir + "/frame_" + std::to_string(frame) + ".vtu";
        cout << "Starting write to " << filename << endl;
        meshes[0].writeToVTK(filename, false);
        cout << "Finishing write to " << filename << endl;
    }

    float total = 0.0f;
    while (++frame < max_frames) {
        string filename = state_output_dir + "/frame_" + std::to_string(frame) + ".vtu";
        cout << "Starting write to " << filename << endl;
        meshes[0].writeToVTK(filename, false);
        cout << "Finishing write to " << filename << endl;

        auto start = std::chrono::high_resolution_clock::now();
        if (is_usingGPU) {
            runStepsGPU();
        }
        else {
            runStepsCPU();
        }
        auto stop = std::chrono::high_resolution_clock::now();
        double step_wallTime = std::chrono::duration<double>(stop - start).count();
        cout << "Frame " << frame << " took " << step_wallTime << " seconds" << endl;
        total += step_wallTime;
    }

    float avg_fps = (max_frames - waiting_frames) / total;
    printf("Total walltime was %f, so avg fps = %f and avg time (ms) = %f\n", total, avg_fps, 1000.0f / avg_fps);
}