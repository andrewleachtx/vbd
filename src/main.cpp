#include "include.h"
#include "include/simulate/PhysicsScene.h"

using std::cout, std::cerr, std::endl;
using std::vector, std::string, std::make_shared, std::shared_ptr, std::pair, std::array, std::tuple;
using std::stoi, std::stoul, std::min, std::max, std::numeric_limits, std::abs;

int main(int argc, char** argv) {
    // Sanitize and parse inputs. We have ./VBD <resource_dir> <scene_no> <state_output_dir> <is_usingGPU=true|false>
    if (argc != 5) {
        cerr << "Usage: ./VBD <resource_dir> <scene_no> <state_output_dir> <is_usingGPU=true|false>" << endl;

        cerr << "You entered: ";
        for (int i = 0; i < argc; i++) {
            cerr << argv[i] << " ";
        }
        cerr << endl;

        return 1;
    }

    // Initialize scene according to parsed information
    string resource_dir = argv[1];
    int scene_no = stoi(argv[2]);
    string state_output_dir = argv[3];
    bool is_usingGPU = (string(argv[4]) == "true");

    PhysicsScene physics(resource_dir, scene_no, state_output_dir, is_usingGPU);
    cout << "[STARTING INITIALIZATION (" << (is_usingGPU ? "GPU" : "CPU") << ")" << endl;
    physics.init();
    cout << "[INITIALIZATION COMPLETE!]" << endl;

    // Start simulation
    cout << "[STARTING SIMULATION]" << endl;
    physics.simulate();
    cout << "[SIMULATION COMPLETE!]" << endl;
}