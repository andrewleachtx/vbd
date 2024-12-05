#include "include.h"

using std::cout, std::cerr, std::endl;
using std::vector, std::string, std::make_shared, std::shared_ptr, std::pair, std::array, std::tuple;
using std::stoi, std::stoul, std::min, std::max, std::numeric_limits, std::abs;

int main(int argc, char** argv) {
    // Sanitize and parse inputs. We have ./VBD <resource_dir> <scene_no> <state_output_dir>
    if (argc != 4) {
        cerr << "Usage: ./VBD <resource_dir> <scene_no> <state_output_dir>" << endl;
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

    PhysicsScene physics(resource_dir, scene_no, state_output_dir);
    cout << "[STARTING INITIALIZATION]" << endl;
    physics.init();
    cout << "[INITIALIZATION COMPLETE!]" << endl;

    // Start simulation
    cout << "[STARTING SIMULATION]" << endl;
    physics.simulate();
    cout << "[SIMULATION COMPLETE!]" << endl;
}