#include "Mesh.h"

#include "../utils/utils.h"

#include <iostream>
#include <string>
#include <vector>
#include <nlohmann/json.hpp>

// https://examples.vtk.org/site/Cxx/
#include <vtk-9.3/vtkSmartPointer.h>
#include <vtk-9.3/vtkUnstructuredGridReader.h>
#include <vtk-9.3/vtkUnstructuredGrid.h>
#include <vtk-9.3/vtkPointData.h>
#include <vtk-9.3/vtkCellData.h>
#include <vtk-9.3/vtkIntArray.h>
#include <vtk-9.3/vtkCellArray.h>
#include <vtk-9.3/vtkTetra.h>

using std::cout, std::cerr, std::endl, std::string, std::vector, std::array, std::ifstream;
using json = nlohmann::json;

void Mesh::initFromJson(const json& mesh_data) {
    mass = mesh_data["params"]["mass"].get<float>();
    mu = mesh_data["params"]["mu"].get<float>();
    lambda = mesh_data["params"]["lambda"].get<float>();
    damping = mesh_data["params"]["damping"].get<float>();
    k_c = mesh_data["params"]["k_c"].get<float>();
    mu_c = mesh_data["params"]["mu_c"].get<float>();
    eps_c = mesh_data["params"]["eps_c"].get<float>();

    for (int i = 0; i < 3; i++) {
        position[i] = mesh_data["params"]["position"][i];
        velocity[i] = mesh_data["params"]["velocity"][i];
    }
}

void Mesh::initFromVTK(const string& vtk_file) {
    auto reader = vtkSmartPointer<vtkUnstructuredGridReader>::New();
    reader->SetFileName(vtk_file.c_str());
    reader->Update();

    // This "grid" has points (vertices) and cells (tetrahedra) which can be accessed through getters
    auto grid = reader->GetOutput();

    // Populate vertices
    const auto& vertices = grid->GetPoints();
    vtkIdType num_vertices = vertices->GetNumberOfPoints();

    // Preallocate to avoid thrashing reallocs, could also resize and use [i] but push_back does same in this context
    init_positions.reserve(num_vertices);
    prev_positions.reserve(num_vertices);
    cur_positions.reserve(num_vertices);
    velocities.reserve(num_vertices);

    cout << "Trying to load " << num_vertices << " vertices" << endl;
    for (vtkIdType i = 0; i < num_vertices; i++) {
        double* vertex = vertices->GetPoint(i);

        // The initial and current / deformable ones are the same at the start - offset by Mesh "pos"
        glm::vec3 pos((float)vertex[0], (float)vertex[1], (float)vertex[2]);
        pos += position;

        init_positions.push_back(pos);
        prev_positions.push_back(pos);
        cur_positions.push_back(pos);

        // Init velocities to whatever the given initial velocity is
        velocities.push_back(velocity);
    }

    // Populate tets
    cout << "Trying to load " << grid->GetNumberOfCells() << " cells" << endl;
    for (vtkIdType i = 0; i < grid->GetNumberOfCells(); i++) { 
        vtkCell* cell = grid->GetCell(i);
        vtkTetra* tet = vtkTetra::SafeDownCast(cell);

        if (tet) {
            std::array<int, 4> tet_idx;
            for (int j = 0; j < 4; j++) {
                tet_idx[j] = tet->GetPointIds()->GetId(j);
            }

            tetrahedra.push_back(tet_idx);
        }
        else {
            cerr << "Failed to get tetrahedra at " << __FILE__ << ": " << __LINE__ << endl;
        }
    }

    // Try to get per vertex colors
    auto vtk_colors = grid->GetPointData()->GetScalars("color");
    cout << "Trying to load colors" << endl;
    if (vtk_colors) {
        assert(vtk_colors->GetNumberOfTuples() == num_vertices);
        colors.reserve(num_vertices);
        for (vtkIdType i = 0; i < vtk_colors->GetNumberOfTuples(); i++) {
            colors.push_back(vtk_colors->GetTuple1(i));
        }

        // Calculate color groups; if our max_color is 0 we have 0 + 1 = max_color + 1 colors
        int max_color = -1;
        for (int i = 0; i < colors.size(); i++) {
            max_color = std::max(max_color, colors[i]);
        }

        // We can identify num_colors as color_groups.size() now
        color_groups.resize(max_color + 1);

        // Populate color groups with the index of each vertice
        for (int i = 0; i < colors.size(); i++) {
            color_groups[colors[i]].push_back(i);
        }

        cout << "Successfully loaded " << max_color + 1 << " colors" << endl;
    }
    else {
        cerr << "Failed to get colors at " << __FILE__ << ": " << __LINE__ << endl;
        return;
    }

    /*
        Because I am nervous about cache inefficiency by using vertex data, we can rearrange
        all the original positions, and if we ever need to know the original we can access
        those with an auxiliary array "old_indices".

        What I mean is if we have colors[0] = [vidx8, vidx343, vidx2382] we are going to
        make very inefficient calls to the main arrays. So instead we can just reorder
        the array by color, such that we really have [new_vidx0, new_vidx1, new_vidx2]

        old_indices[new_idx] = old_idx, so we have a way back if we need it
    */
    vector<glm::vec3> new_positions, new_velocities;
    vector<int> new_colors;
    old_indices.resize(num_vertices);

    size_t idx = 0;
    cout << "Reordering vertices based on color" << endl;
    for (int color = 0; color < color_groups.size(); color++) {
        for (int i = 0; i < color_groups[color].size(); i++) {
            int old_idx = color_groups[color][i];
            old_indices[old_idx] = idx;

            new_positions.push_back(init_positions[old_idx]);
            new_velocities.push_back(velocities[old_idx]);
            new_colors.push_back(colors[old_idx]);

            idx++;
        }
    }

    // Swap the old arrays for the new ones
    init_positions = std::move(new_positions);
    velocities = std::move(new_velocities);
    colors = std::move(new_colors);
        
    // Sanity check
    assert(init_positions.size() == num_vertices);
    assert(velocities.size() == num_vertices);
    assert(colors.size() == num_vertices);
    assert(old_indices.size() == num_vertices);
}