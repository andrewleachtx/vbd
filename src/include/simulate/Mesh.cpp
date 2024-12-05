#include "Mesh.h"

#include "../utils/utils.h"

#include <iostream>
#include <string>
#include <vector>
#include <nlohmann/json.hpp>

// https://examples.vtk.org/site/Cxx/
#include <vtk-9.3/vtkSmartPointer.h>
#include <vtk-9.3/vtkUnstructuredGridReader.h>
#include <vtk-9.3/vtkXMLUnstructuredGridWriter.h>
#include <vtk-9.3/vtkUnstructuredGridWriter.h>
#include <vtk-9.3/vtkUnstructuredGrid.h>
#include <vtk-9.3/vtkPointData.h>
#include <vtk-9.3/vtkCellData.h>
#include <vtk-9.3/vtkIntArray.h>
#include <vtk-9.3/vtkCellArray.h>
#include <vtk-9.3/vtkTetra.h>

using std::cout, std::cerr, std::endl, std::string, std::vector, std::array, std::ifstream;
using json = nlohmann::json;

glm::vec3 Mesh::computeEnergyFirstOrder(size_t v_idx, size_t tet_idx) { return glm::vec3(0.0f); }
glm::mat3 Mesh::computeEnergySecondOrder(size_t v_idx, size_t tet_idx) { return glm::mat3(0.0f); }

void Mesh::writeToVTK(const string& output_dir, bool raw) {
    vtkSmartPointer<vtkPoints> vtk_points = vtkSmartPointer<vtkPoints>::New();
    for (const auto& pos : cur_positions) {
        vtk_points->InsertNextPoint(pos.x, pos.y, pos.z);
    }

    vtkSmartPointer<vtkCellArray> vtk_cells = vtkSmartPointer<vtkCellArray>::New();
    for (const auto& tet : tetrahedra) {
        vtkSmartPointer<vtkTetra> vtkTet = vtkSmartPointer<vtkTetra>::New();
        vtkTet->GetPointIds()->SetId(0, tet[0]);
        vtkTet->GetPointIds()->SetId(1, tet[1]);
        vtkTet->GetPointIds()->SetId(2, tet[2]);
        vtkTet->GetPointIds()->SetId(3, tet[3]);
        vtk_cells->InsertNextCell(vtkTet);
    }
    
    vtkSmartPointer<vtkUnstructuredGrid> vtk_mesh = vtkSmartPointer<vtkUnstructuredGrid>::New();
    vtk_mesh->SetPoints(vtk_points);
    vtk_mesh->SetCells(VTK_TETRA, vtk_cells);

    if (raw) {
        vtkSmartPointer<vtkUnstructuredGridWriter> raw_writer = vtkSmartPointer<vtkUnstructuredGridWriter>::New();
        raw_writer->SetFileName(output_dir.substr(0, output_dir.size() - 4).append("_raw.vtu").c_str());
        raw_writer->SetInputData(vtk_mesh);
        raw_writer->Write();
    }
    else {
        vtkSmartPointer<vtkXMLUnstructuredGridWriter> writer = vtkSmartPointer<vtkXMLUnstructuredGridWriter>::New();
        writer->SetFileName(output_dir.c_str());
        writer->SetInputData(vtk_mesh);
        writer->Write();
    }
}

/*
    f_i = - (m_i / h^2) * (xi - yi) - SUM_j (d(E_j(x)) / d(x_i))

    First term is inertia, second elastic
*/
glm::vec3 Mesh::computeForces(size_t v_idx, float dt) {
    // We can assume uniform for now, but not ideal TODO: add per vertex mass buffer
    //float m_i = mass / (float)cur_positions.size();
    float m_i = mass;
    glm::vec3 f_i_inertia = (m_i / (dt * dt)) * (cur_positions[v_idx] - y[v_idx]);
    
    // Iterate over nearby tetrahedra (ones that are attached to vertex v_idx) for elastic forces
    glm::vec3 f_i_elastic(0.0f);

    const vector<int>& neighbors = vertex2tets[v_idx];
    for (size_t tet_idx : neighbors) {
        f_i_elastic += computeEnergyFirstOrder(v_idx, tet_idx);
    }

    return -f_i_inertia - f_i_elastic;
}

glm::mat3 Mesh::computeHessian(size_t v_idx, float dt) {
    //float m_i = mass / (float)cur_positions.size();
    float m_i = mass;
    glm::mat3 H_i_inertia = (m_i / (dt * dt)) * glm::mat3(1.0f);

    // Second term is the sum of Hessians of the force elements wrt v_idx
    glm::mat3 H_i_elastic(0.0f);

    const vector<int>& neighbors = vertex2tets[v_idx];
    for (size_t tet_idx : neighbors) {
        // TODO: If this is too complex, use an approximation of energy = (mu + lambda) * V0
        H_i_elastic += computeEnergySecondOrder(v_idx, tet_idx);
        // H_i_elastic += (mu + lambda) * tet_volumes[tet_idx] * glm::mat3(1.0f);
    }

    return H_i_inertia + H_i_elastic;
}

void Mesh::doVBDCPU(float dt) {
    // Could substep this if needed?

    // Iterate over each vertex per color
    vector<glm::vec3> x_new(cur_positions.size());
    for (size_t c = 0; c < color_ranges.size(); c++) { // lol c++
        size_t start(color_ranges[c][0]), end(color_ranges[c][1]);

        #pragma omp parallel for
        for (size_t i = start; i < end; i++) {
            // f_i is the total forces acting on the vertex
            glm::vec3 f_i = computeForces(i, dt);

            // H_i is the Hessian of G_i 
            glm::mat3 H_i = computeHessian(i, dt);

            // Solve for delta_x_i = -H_i^-1 * f_i
            float det_H_i = glm::determinant(H_i);
            if (std::abs(det_H_i) < FLOAT_EPS) {
                H_i += FLOAT_EPS * glm::mat3(1.0f);
            }

            glm::vec3 delta_x_i = -glm::inverse(H_i) * f_i;
            
            // Perform optional line search, for now idk
            x_new[i] = cur_positions[i] + delta_x_i;
        }

        // Update the positions
        #pragma omp parallel for
        for (int i = start; i < end; i++) {
            cur_positions[i] = x_new[i];
        }
    }
}

/*
    The paper presents four options,

    a) Previous position
    b) Inertia
    c) Inertia and acceleration
    d) Adaptive initialization with a~, store in x aka cur_position

    After some struggle getting adaptive acceleration working, I have decided to leave it
    but as a parameter of "initGuessType" to choose what you use.
*/
void Mesh::initialGuess(float dt, const glm::vec3& a_ext) {
    size_t num_vertices = init_positions.size();

    if (initGuessType == initGuessEnum::ADAPTIVE) {
        float a_ext_mag = glm::length(a_ext);
        glm::vec3 a_ext_hat = a_ext / a_ext_mag;

        // We need to compute the initial guess for each vertex
        for (size_t i = 0; i < num_vertices; i++) {
            // a_t = (v_t - v_t-1) / h and compute component along external acceleration direction
            // TODO: Formula 6 suggests we don't need a buffer for cur and prev velocities
            glm::vec3 a_t = (cur_velocities[i] - prev_velocities[i]) / dt;
            float a_t_ext = glm::dot(a_t, a_ext_hat);

            // Update previous pos
            prev_positions[i] = cur_positions[i];

            float a_coef;
            if (a_t_ext > a_ext_mag) {
                a_coef = 1.0f;
            }
            else if (a_t_ext < 0.0f) {
                a_coef = 0.0f;
            }
            else {
                a_coef = a_t_ext / a_ext_mag;
            }

            glm::vec3 a_tilde = a_coef * a_ext;

            // x = x_t + hv_t + h^2(a~)
            cur_positions[i] += (dt * cur_velocities[i]) + (dt * dt * a_tilde);
            y[i] = cur_positions[i];
        }
    }
    else if (initGuessType == initGuessEnum::INERTIA_ACCEL) {
        for (size_t i = 0; i < num_vertices; i++) {
            // y = x = x_t + hv_t + h^2a_ext 
            cur_positions[i] = prev_positions[i] + (dt * cur_velocities[i]) + (dt * dt * a_ext);
            y[i] = cur_positions[i];
            
            // Update previous pos
            prev_positions[i] = cur_positions[i];
        }
    }
}

void Mesh::updateVelocities(float dt) {
    for (size_t i = 0; i < cur_velocities.size(); i++) {
        prev_velocities[i] = cur_velocities[i];

        cur_velocities[i] = (cur_positions[i] - prev_positions[i]) / dt;
    }
}

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

    // Assume initial guess type is always inertia + accel
    initGuessType = initGuessEnum::INERTIA_ACCEL;

    cout << "Parameters:" << endl;
    cout << "\tmass: " << mass << endl;
    cout << "\tmu: " << mu << endl;
    cout << "\tlambda: " << lambda << endl;
    cout << "\tdamping: " << damping << endl;
    cout << "\tk_c: " << k_c << endl;
    cout << "\tmu_c: " << mu_c << endl;
    cout << "\teps_c: " << eps_c << endl;
    cout << "\tposition: " << position.x << " " << position.y << " " << position.z << endl;
    cout << "\tvelocity: " << velocity.x << " " << velocity.y << " " << velocity.z << endl;
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
    prev_velocities.reserve(num_vertices);
    cur_velocities.reserve(num_vertices);
    y.reserve(num_vertices);

    cout << "Trying to load " << num_vertices << " vertices" << endl;
    for (vtkIdType i = 0; i < num_vertices; i++) {
        double* vertex = vertices->GetPoint(i);

        // The initial and current / deformable ones are the same at the start - offset by Mesh "pos"
        glm::vec3 pos((float)vertex[0], (float)vertex[1], (float)vertex[2]);
        pos += position;

        init_positions.push_back(pos);

        // Init velocities to whatever the given initial velocity is
        prev_velocities.push_back(velocity);
        cur_velocities.push_back(velocity);
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

    std::vector<std::vector<int>> color_groups;

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

        Incase we need it
            old_indices[new_idx] = old_idx
            new_indices[old_idx] = new_idx
    */
    vector<glm::vec3> new_positions, new_velocities;
    vector<int> new_colors;
    vector<int> old2new(num_vertices);

    new_positions.reserve(num_vertices);
    new_velocities.reserve(num_vertices);
    new_colors.reserve(num_vertices);

    size_t idx = 0;
    cout << "Reordering vertices based on color for cache efficiency" << endl;
    for (int color = 0; color < color_groups.size(); color++) {
        for (int i = 0; i < color_groups[color].size(); i++) {
            int old_idx = color_groups[color][i];

            old2new[old_idx] = idx;

            new_positions.push_back(init_positions[old_idx]);
            new_velocities.push_back(cur_velocities[old_idx]);
            new_colors.push_back(colors[old_idx]);

            idx++;
        }
    }

    // Each vertex of a tetrahedra is old; we need remap
    for (int i = 0; i < tetrahedra.size(); i++) {
        for (int j = 0; j < 4; j++) {
            tetrahedra[i][j] = old2new[tetrahedra[i][j]];
        }
    }

    // We need to store one way access to all tetrahedra indices given a vertex for quick access to neighbors
    vertex2tets.resize(num_vertices);
    for (int i = 0; i < tetrahedra.size(); i++) {
        for (int j = 0; j < 4; j++) {
            vertex2tets[tetrahedra[i][j]].push_back(i);
        }
    }

    // Precalculate relevant tetrahedron data
    Dm_inverses.resize(tetrahedra.size());
    tet_volumes.resize(tetrahedra.size());
    for (int i = 0; i < tetrahedra.size(); i++) {
        const std::array<int, 4>& tet = tetrahedra[i];

        glm::vec3 x0 = init_positions[tet[0]];
        glm::vec3 x1 = init_positions[tet[1]];
        glm::vec3 x2 = init_positions[tet[2]];
        glm::vec3 x3 = init_positions[tet[3]];

        // Dm is the reference shape matrix, which is defined by the three edge vectors
        glm::mat3 Dm;
        Dm[0] = x1 - x0;
        Dm[1] = x2 - x0;
        Dm[2] = x3 - x0;

        Dm_inverses[i] = glm::inverse(Dm);

        float vol = glm::determinant(Dm) / 6.0f;
        // if (fabsf(vol) < FLOAT_EPS) {
        //     cout << "Degenerate tetrahedron " << i << " has zero volume, exiting" << endl;
        //     throw std::runtime_error("Degenerate tetrahedron detected");
        // }
        tet_volumes[i] = vol;
    }

    // Update prev_pos, cur_pos, and y positions
    prev_positions = new_positions;
    cur_positions = new_positions;
    y = new_positions;

    // Swap the old arrays for the new ones
    init_positions = std::move(new_positions);
    cur_velocities = std::move(new_velocities);
    colors = std::move(new_colors);

    // For fun we can evaluate density
    vector<float> color_freq(color_groups.size());
    for (int i = 0; i < color_groups.size(); i++) {
        color_freq[i] = (float)color_groups[i].size();
    }

    cout << "Ideal vertices per group = " << ((float)num_vertices / color_groups.size()) << " versus actual: ";
    for (int i = 0; i < color_freq.size(); i++) {
        cout << color_freq[i] << " ";
    }
    cout << endl;

    // Calculate color ranges. This is coming along so nicely!
    size_t start_idx = 0;
    color_ranges.reserve(color_groups.size());
    for (int color = 0; color < color_groups.size(); color++) {
        size_t group_sz = color_groups[color].size();
        color_ranges.push_back({start_idx, start_idx + group_sz});
        start_idx += group_sz;
    }

    // Print color ranges
    for (int i = 0; i < color_ranges.size(); i++) {
        cout << "\tColor " << i << ": " << "[" << color_ranges[i][0] << ", " << color_ranges[i][1] << ")" << endl;
    }

    // Sanity check
    assert(init_positions.size() == num_vertices);
    assert(prev_velocities.size() == num_vertices);
    assert(colors.size() == num_vertices);
}