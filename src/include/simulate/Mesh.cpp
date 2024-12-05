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

/*
    Helper for the elastic portion of the local energy first order calculation, which is the second
    part of equation (8)
*/
glm::vec3 Mesh::computeEnergyFirstOrder(size_t v_idx, size_t tet_idx) {
    const std::array<int, 4>& tet = tetrahedra[tet_idx];

    glm::vec3 x0 = cur_positions[tet[0]];
    glm::vec3 x1 = cur_positions[tet[1]];
    glm::vec3 x2 = cur_positions[tet[2]];
    glm::vec3 x3 = cur_positions[tet[3]];

    // F = Ds * Dm_inv where Ds is the current and deformed shape matrix and Dm_inv is the inverse of the rest state
    glm::mat3 Ds;
    Ds[0] = x1 - x0;
    Ds[1] = x2 - x0;
    Ds[2] = x3 - x0;

    const glm::mat3& Dm_inv = Dm_inverses[tet_idx];
    glm::mat3 F = Ds * Dm_inv;

    // Piola-Kirchhoff stress tensor P = mu * (F - F_it) + lm * (log(J) * FinvT)
    glm::mat3 F_it = glm::transpose(glm::inverse(F));
    float J = glm::determinant(F);

    glm::mat3 P = mu * (F - F_it) + lambda * (log(J) * F_it);

    /*
        Elastic force at the ith vertice is fi = -V_0 * (P * grad_N_i) where

        N_i represents the shape function that interpolates displacement in the tetrahedron. We can
        use grad(N_i) to get the gradient of the shape function. Basically, when the shape is very
        deformed, the force will be very large and accentuated in the direction of the deformation.

        //TODO: There is also another way to implement this, see VBD_NeoHookean.h/inline void computeElasticEnergy(int tetId, T& energy)
    */

    glm::mat3 Dm_it = glm::transpose(Dm_inv);
    float V0 = tet_volumes[tet_idx];

    // !FIXME: This is pretty awful, but only way to work with my input, please change
    glm::vec3 grad_N0 = Dm_it[0] - Dm_it[1] - Dm_it[2];
    glm::vec3 grad_N1 = Dm_it[0];
    glm::vec3 grad_N2 = Dm_it[1];
    glm::vec3 grad_N3 = Dm_it[2];
    
    glm::vec3 f0 = -V0 * (P * grad_N0);
    glm::vec3 f1 = -V0 * (P * grad_N1);
    glm::vec3 f2 = -V0 * (P * grad_N2);
    glm::vec3 f3 = -V0 * (P * grad_N3);

    if (v_idx == tet[0]) {
        return f0;
    }
    else if (v_idx == tet[1]) {
        return f1;
    }
    else if (v_idx == tet[2]) {
        return f2;
    }
    else if (v_idx == tet[3]) {
        return f3;
    }

    return glm::vec3(0.0f);
}

/*
    This is specifically to be used as for the elastic energy calculation summation
    in the Hessian computation, or the second part of equation (9)
*/
glm::mat3 Mesh::computeEnergySecondOrder(size_t v_idx, size_t tet_idx) {
    const std::array<int, 4>& tet = tetrahedra[tet_idx];

    glm::vec3 x0 = cur_positions[tet[0]];
    glm::vec3 x1 = cur_positions[tet[1]];
    glm::vec3 x2 = cur_positions[tet[2]];
    glm::vec3 x3 = cur_positions[tet[3]];

    // F = Ds * Dm_inv where Ds is the current and deformed shape matrix and Dm_inv is the inverse of the rest state
    glm::mat3 Ds;
    Ds[0] = x1 - x0;
    Ds[1] = x2 - x0;
    Ds[2] = x3 - x0;

    const glm::mat3& Dm_inv = Dm_inverses[tet_idx];
    glm::mat3 F = Ds * Dm_inv;

    // Piola-Kirchhoff stress tensor P = mu * (F - F_it) + lm * (log(J) * FinvT)
    glm::mat3 F_it = glm::transpose(glm::inverse(F));
    float J = glm::determinant(F);
    glm::mat3 P = mu * (F - F_it) + lambda * (log(J) * F_it);

    glm::mat3 Dm_it = glm::transpose(Dm_inv);
    float V0 = tet_volumes[tet_idx];

    glm::vec3 grad_N0 = Dm_it[0] - Dm_it[1] - Dm_it[2];
    glm::vec3 grad_N1 = Dm_it[0];
    glm::vec3 grad_N2 = Dm_it[1];
    glm::vec3 grad_N3 = Dm_it[2];

    // Compute material tangent stiffness tensor C (9x9 matrix as it is 3x3 for each vertex)
    glm::mat3 dF_dxi;
    if (v_idx == tet[0]) {
        dF_dxi = -glm::outerProduct(glm::vec3(1.0f), grad_N0);
    }
    else if (v_idx == tet[1]) {
        dF_dxi = glm::outerProduct(glm::vec3(1.0f), grad_N1);
    }
    else if (v_idx == tet[2]) {
        dF_dxi = glm::outerProduct(glm::vec3(1.0f), grad_N2);
    }
    else if (v_idx == tet[3]) {
        dF_dxi = glm::outerProduct(glm::vec3(1.0f), grad_N3);
    }

    // Compute adjugate of F
    glm::mat3 adjF = glm::transpose(F);

    // Flatten adjF into ddetF_dF
    float ddetF_dF[9] = {
        adjF[0][0], adjF[1][0], adjF[2][0],
        adjF[0][1], adjF[1][1], adjF[2][1],
        adjF[0][2], adjF[1][2], adjF[2][2]
    };

    // Initialize C
    float C[9][9] = {0};

    // Compute C = lambda * (ddetF_dF * ddetF_dF^T)
    for (int i = 0; i < 9; ++i) {
        for (int j = 0; j < 9; ++j) {
            C[i][j] = lambda * ddetF_dF[i] * ddetF_dF[j];
        }
    }

    // Add mu to diagonal elements
    for (int i = 0; i < 9; ++i) {
        C[i][i] += mu;
    }

    // H_i = d(F)/d(x_i)^T * ( C * d(F)/d(x_i) )
    // Flatten dF_dxi into a 9-element vector
    float dF_dxi_vec9[9] = {
        dF_dxi[0][0], dF_dxi[1][0], dF_dxi[2][0],
        dF_dxi[0][1], dF_dxi[1][1], dF_dxi[2][1],
        dF_dxi[0][2], dF_dxi[1][2], dF_dxi[2][2]
    };

    // Compute CdF_dxi = C * dF_dxi_vec9
    float CdF_dxi[9] = {0};
    for (int i = 0; i < 9; ++i) {
        for (int j = 0; j < 9; ++j) {
            CdF_dxi[i] += C[i][j] * dF_dxi_vec9[j];
        }
    }

    // Compute H_i_scalar = dF_dxi_vec9^T * CdF_dxi
    float H_i_scalar = 0.0f;
    for (int i = 0; i < 9; ++i) {
        H_i_scalar += dF_dxi_vec9[i] * CdF_dxi[i];
    }

    H_i_scalar *= V0;

    glm::mat3 H_i_elastic = H_i_scalar * glm::mat3(1.0f);

    return H_i_elastic;
}

/*
    f_i = - (m_i / h^2) * (xi - yi) - SUM_j (d(E_j(x)) / d(x_i))

    First term is inertia, second elastic
*/
glm::vec3 Mesh::computeForces(size_t v_idx, float dt) {
    // We can assume uniform for now, but not ideal TODO: add per vertex mass buffer
    float m_i = mass / (float)cur_positions.size();
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
    float m_i = mass / (float)cur_positions.size();
    glm::mat3 H_i_inertia = (m_i / (dt * dt)) * glm::mat3(1.0f);

    // Second term is the sum of Hessians of the force elements wrt v_idx
    glm::mat3 H_i_elastic(0.0f);

    const vector<int>& neighbors = vertex2tets[v_idx];
    for (size_t tet_idx : neighbors) {
        // TODO: If this is too complex, use an approximation of energy = (mu + lambda) * V0
        H_i_elastic += computeEnergySecondOrder(v_idx, tet_idx);
    }

    return H_i_inertia + H_i_elastic;
}

void Mesh::doVBDCPU(float dt) {
    // Could substep this if needed?

    // Iterate over each vertex per color
    vector<glm::vec3> x_new(cur_positions.size());
    for (int c = 0; c < color_ranges.size(); c++) { // lol c++
        size_t start(color_ranges[c][0]), end(color_ranges[c][1]);

        cout << "Starting loop with color range " << start << " to " << end << endl;
        #pragma omp parallel for
        for (int i = start; i < end; i++) {
            // f_i is the total forces acting on the vertex
            glm::vec3 f_i = computeForces(i, dt);

            // H_i is the Hessian of G_i 
            glm::mat3 H_i = computeHessian(i, dt);

            // Solve for delta_x_i = -H_i^-1 * f_i
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
    Adaptive initialization with a~, store in x aka cur_position
*/
void Mesh::initialGuessAdaptive(float dt, const glm::vec3& a_ext) {
    size_t num_vertices = init_positions.size();

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

        float a_tilde = a_t_ext;
        if (a_t_ext < 0.0f) {
            a_tilde = 0.0f;
        }
        else if (a_t_ext > -FLOAT_EPS && a_t_ext < FLOAT_EPS) {
            a_tilde *= a_t_ext / a_ext_mag;
        }

        // x = x_t + hv_t + h^2(a~)
        cur_positions[i] = (cur_positions[i]) + (dt * cur_velocities[i]) + (dt * dt * a_tilde);
    }

    // This is a little weird, but we can store this initial guess in y
    y = cur_positions;
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
        cout << "Color " << i << ": " << "[" << color_ranges[i][0] << ", " << color_ranges[i][1] << ")" << endl;
    }

    // Sanity check
    assert(init_positions.size() == num_vertices);
    assert(prev_velocities.size() == num_vertices);
    assert(colors.size() == num_vertices);
}