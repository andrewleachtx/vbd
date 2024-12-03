#include "Mesh.h"

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
    mass = mesh_data["params"]["mass"];
    mu = mesh_data["params"]["mu"];
    lambda = mesh_data["params"]["lambda"];
    damping = mesh_data["params"]["damping"];
    k_c = mesh_data["params"]["k_c"];
    mu_c = mesh_data["params"]["mu_c"];
    eps_c = mesh_data["params"]["eps_c"];

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
    const auto vertices& = grid->GetPoints();
    printf("Attempting to load %d vertices\n", vertices->GetNumberOfPoints());
    for (size_t i = 0; i < vertices->GetNumberOfPoints(); i++) {
        double* vertex = vertices->GetPoint(i);

        // The initial and current / deformable ones are the same at the start
        initpos_x.push_back((float)vertex[0]);
        initpos_y.push_back((float)vertex[1]);
        initpos_z.push_back((float)vertex[2]);

        pos_x.push_back((float)vertex[0]);
        pos_y.push_back((float)vertex[1]);
        pos_z.push_back((float)vertex[2]);
    }

    // Populate tets
    printf("Attempting to load %d cells\n", grid->GetCells()->GetNumberOfCells());
    const auto cells = grid->GetCells();
    for (size_t i = 0; i < cells->GetNumberOfCells(); i++) {
        // Attempt to get the tetrahedra
        vtkTetra* tet = vtkTetra::SafeDownCast(cells->GetCell(i));

        if (tet) {
            int* indices = tet->GetPointIds()->GetPointer(0);

            tet_v1.push_back(indices[0]);
            tet_v2.push_back(indices[1]);
            tet_v3.push_back(indices[2]);
            tet_v4.push_back(indices[3]);
        }
        else {
            cerr << "Failed to get tetrahedra at " << __FILE__ << ": " << __LINE__ << endl;
        }
    }
}