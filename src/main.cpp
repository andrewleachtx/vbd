#include "include.h"

using std::cout, std::cerr, std::endl;
using std::vector, std::string, std::make_shared, std::shared_ptr, std::pair, std::array, std::tuple;
using std::stoi, std::stoul, std::min, std::max, std::numeric_limits, std::abs;

int main(int argc, char** argv) {
    // Sanitize and parse inputs. We have ./VBD <scene_no> <state_output_dir>
    if (argc != 3) {
        cerr << "Usage: ./VBD <scene_no> <state_output_dir>" << endl;
        return 1;
    }

    // Initialize scene according to parsed information

    // Start simulation 

}