#include "Mesh.h"

#include "../utils/utils.h"

#include <iostream>
#include <string>
#include <vector>
#include <nlohmann/json.hpp>
#include <Eigen/Dense>
#include <Eigen/IterativeLinearSolvers>

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

#include <TinyAD/VectorFunction.hh>
#include <TinyAD/Utils/NewtonDecrement.hh>
#include <TinyAD/ScalarFunction.hh>
#include <TinyAD/Scalar.hh>

using std::cout, std::cerr, std::endl, std::string, std::vector, std::array, std::ifstream;
using json = nlohmann::json;

void Mesh::assembleVertexVForceAndHessian(const Eigen::Matrix<float, 9, 1>& dE_dF,
                                    const Eigen::Matrix<float, 9, 9>& d2E_dF_dF,
                                    float m1, float m2, float m3,
                                    Eigen::Vector3f& force, Eigen::Matrix3f& h) {
    float A1 = dE_dF(0);
    float A2 = dE_dF(1);
    float A3 = dE_dF(2);
    float A4 = dE_dF(3);
    float A5 = dE_dF(4);
    float A6 = dE_dF(5);
    float A7 = dE_dF(6);
    float A8 = dE_dF(7);
    float A9 = dE_dF(8);

    force << A1 * m1 + A4 * m2 + A7 * m3,
             A2 * m1 + A5 * m2 + A8 * m3,
             A3 * m1 + A6 * m2 + A9 * m3;
    
    Eigen::Matrix<float, 3, 9> HL;

    HL.row(0) = d2E_dF_dF.row(0) * m1 + d2E_dF_dF.row(3) * m2 + d2E_dF_dF.row(6) * m3;
	HL.row(1) = d2E_dF_dF.row(1) * m1 + d2E_dF_dF.row(4) * m2 + d2E_dF_dF.row(7) * m3;
	HL.row(2) = d2E_dF_dF.row(2) * m1 + d2E_dF_dF.row(5) * m2 + d2E_dF_dF.row(8) * m3;

	h.col(0) = HL.col(0) * m1 + HL.col(3) * m2 + HL.col(6) * m3;
	h.col(1) = HL.col(1) * m1 + HL.col(4) * m2 + HL.col(7) * m3;
	h.col(2) = HL.col(2) * m1 + HL.col(5) * m2 + HL.col(8) * m3;
}

void Mesh::computeElasticEnergyGradients(float dt, size_t v_idx, size_t tet_idx,
                                         Eigen::Vector3f& force, Eigen::Matrix3f& hessian) {
    const std::array<int, 4> tet = tetrahedra[tet_idx];
    float a = 1 + mu / lambda;
    float A = tet_volumes[tet_idx];    
    Eigen::Vector3f x0 = cur_positions[tet[0]];
    Eigen::Vector3f x1 = cur_positions[tet[1]];
    Eigen::Vector3f x2 = cur_positions[tet[2]];
    Eigen::Vector3f x3 = cur_positions[tet[3]];

    Eigen::Matrix3f Dm_inv = Dm_inverses[tet_idx];
    Eigen::Matrix3f Ds;
    Ds.col(0) = x1 - x0;
    Ds.col(1) = x2 - x0;
    Ds.col(2) = x3 - x0;
    Eigen::Matrix3f F = Ds * Dm_inv;

    float detF = F.determinant();

    Eigen::Map<Eigen::Matrix<float, 9, 1>> dPhi_D_dF(F.data());

    float F1_1 = F(0, 0);
    float F2_1 = F(1, 0);
    float F3_1 = F(2, 0);
    float F1_2 = F(0, 1);
    float F2_2 = F(1, 1);
    float F3_2 = F(2, 1);
    float F1_3 = F(0, 2);
    float F2_3 = F(1, 2);
    float F3_3 = F(2, 2);

    Eigen::Matrix<float, 9, 1> ddetF_dF;
    ddetF_dF << F2_2 * F3_3 - F2_3 * F3_2,
                F1_3* F3_2 - F1_2 * F3_3,
                F1_2* F2_3 - F1_3 * F2_2,
                F2_3* F3_1 - F2_1 * F3_3,
                F1_1* F3_3 - F1_3 * F3_1,
                F1_3* F2_1 - F1_1 * F2_3,
                F2_1* F3_2 - F2_2 * F3_1,
                F1_2* F3_1 - F1_1 * F3_2,
                F1_1* F2_2 - F1_2 * F2_1;

    Eigen::Matrix<float, 9, 9> d2E_dF_dF = ddetF_dF * ddetF_dF.transpose();

    float k = detF - a;
    d2E_dF_dF(0, 4) += k * F3_3;
    d2E_dF_dF(4, 0) += k * F3_3;
    d2E_dF_dF(0, 5) += k * -F2_3;
    d2E_dF_dF(5, 0) += k * -F2_3;
    d2E_dF_dF(0, 7) += k * -F3_2;
    d2E_dF_dF(7, 0) += k * -F3_2;
    d2E_dF_dF(0, 8) += k * F2_2;
    d2E_dF_dF(8, 0) += k * F2_2;

    d2E_dF_dF(1, 3) += k * -F3_3;
    d2E_dF_dF(3, 1) += k * -F3_3;
    d2E_dF_dF(1, 5) += k * F1_3;
    d2E_dF_dF(5, 1) += k * F1_3;
    d2E_dF_dF(1, 6) += k * F3_2;
    d2E_dF_dF(6, 1) += k * F3_2;
    d2E_dF_dF(1, 8) += k * -F1_2;
    d2E_dF_dF(8, 1) += k * -F1_2;

    d2E_dF_dF(2, 3) += k * F2_3;
    d2E_dF_dF(3, 2) += k * F2_3;
    d2E_dF_dF(2, 4) += k * -F1_3;
    d2E_dF_dF(4, 2) += k * -F1_3;
    d2E_dF_dF(2, 6) += k * -F2_2;
    d2E_dF_dF(6, 2) += k * -F2_2;
    d2E_dF_dF(2, 7) += k * F1_2;
    d2E_dF_dF(7, 2) += k * F1_2;

    d2E_dF_dF(3, 7) += k * F3_1;
    d2E_dF_dF(7, 3) += k * F3_1;
    d2E_dF_dF(3, 8) += k * -F2_1;
    d2E_dF_dF(8, 3) += k * -F2_1;

    d2E_dF_dF(4, 6) += k * -F3_1;
    d2E_dF_dF(6, 4) += k * -F3_1;
    d2E_dF_dF(4, 8) += k * F1_1;
    d2E_dF_dF(8, 4) += k * F1_1;

    d2E_dF_dF(5, 6) += k * F2_1;
    d2E_dF_dF(6, 5) += k * F2_1;
    d2E_dF_dF(5, 7) += k * -F1_1;
    d2E_dF_dF(7, 5) += k * -F1_1;

    d2E_dF_dF *= lambda;

    // Or d2E_dF_dF += mu * I_9x9

    d2E_dF_dF(0, 0) += mu;
    d2E_dF_dF(1, 1) += mu;
    d2E_dF_dF(2, 2) += mu;
    d2E_dF_dF(3, 3) += mu;
    d2E_dF_dF(4, 4) += mu;
    d2E_dF_dF(5, 5) += mu;
    d2E_dF_dF(6, 6) += mu;
    d2E_dF_dF(7, 7) += mu;
    d2E_dF_dF(8, 8) += mu;

    Eigen::Matrix<float, 9, 1> dE_dF = A * (mu * dPhi_D_dF + lambda * (detF - a) * ddetF_dF);

    float DmInv1_1 = Dm_inv(0, 0);
    float DmInv2_1 = Dm_inv(1, 0);
    float DmInv3_1 = Dm_inv(2, 0);
    float DmInv1_2 = Dm_inv(0, 1);
    float DmInv2_2 = Dm_inv(1, 1);
    float DmInv3_2 = Dm_inv(2, 1);
    float DmInv1_3 = Dm_inv(0, 2);
    float DmInv2_3 = Dm_inv(1, 2);
    float DmInv3_3 = Dm_inv(2, 2);

    // TODO: I don't think I need this?
    // int vertedTetVId = getVertexNeighborTetVertexOrder(v_idx, tet_idx);

    // I believe this is calculated from https://animation.rwth-aachen.de/media/papers/2014-CAG-PBER.pdf
    Eigen::Matrix<float, 9, 3> dF_dxi;
    Eigen::Vector3f dE_dxi;
    Eigen::Matrix3f d2E_dxi_dxi;
    float m1, m2, m3;

    if (v_idx == tet[0]) {
        m1 = -DmInv1_1 - DmInv2_1 - DmInv3_1;
        m2 = -DmInv1_2 - DmInv2_2 - DmInv3_2;
        m3 = -DmInv1_3 - DmInv2_3 - DmInv3_3;
    }
    else if (v_idx == tet[1]) {
        m1 = DmInv1_1;
        m2 = DmInv1_2;
        m3 = DmInv1_3;
    }
    else if (v_idx == tet[2]) {
        m1 = DmInv2_1;
        m2 = DmInv2_2;
        m3 = DmInv2_3;
    }
    else if (v_idx == tet[3]) {
        m1 = DmInv3_1;
        m2 = DmInv3_2;
        m3 = DmInv3_3;
    }

    assembleVertexVForceAndHessian(dE_dF, d2E_dF_dF, m1, m2, m3, dE_dxi, d2E_dxi_dxi);

    /*
        Damping term

        f_i -= (kd / h) * (d2E_dxi_dxi) * (x_i - xt_i)
        H_i += (kd / h) * (d2E_dxi_dxi)
    */

    // auto f_i_damping = (damping / dt) * (d2E_dxi_dxi * (cur_positions[v_idx] - y[v_idx]));
    // auto H_i_damping = (damping / dt) * d2E_dxi_dxi;
    auto f_i_damping = Eigen::Vector3f::Zero();
    auto H_i_damping = Eigen::Matrix3f::Zero();

    force -= (dE_dxi + f_i_damping);
    hessian += (d2E_dxi_dxi + H_i_damping);
}

void Mesh::doVBDCPU(float dt) {
    float inv_dt = 1.0f / dt;
    float inv_dtdt = inv_dt * inv_dt;
    size_t num_vertices = cur_positions.size();
    vector<Eigen::Vector3f> x_new(num_vertices, Eigen::Vector3f::Zero());
    float m_i = mass;

    // Per color group batch now
    for (size_t c = 0; c < color_ranges.size(); c++) { // lol c++
        size_t start(color_ranges[c][0]), end(color_ranges[c][1]);

        #pragma omp parallel for
        for (size_t i = start; i < end; i++) {
            /*
                f_i (8) and H_i (9)
            */

            Eigen::Vector3f f_i = - (m_i * inv_dtdt) * (cur_positions[i] - y[i]);
            Eigen::Matrix3f H_i = (m_i * inv_dtdt) * Eigen::Matrix3f::Identity();

            // Accumulate elastic contributions from all tetrahedra neighboring the current vertex
            // Eigen::Vector3f f_i_elastic = Eigen::Vector3f::Zero();
            // Eigen::Matrix3f H_i_elastic = Eigen::Matrix3f::Zero();

            const vector<int>& neighbors = vertex2tets[i];
            for (const int& tet_idx : neighbors) {
                computeElasticEnergyGradients(dt, i, tet_idx, f_i, H_i);
            }

            // f_i += f_i_elastic;
            // H_i += H_i_elastic;

            if (H_i.determinant() > FLOAT_EPS) {
                const Eigen::Vector3f delta_xi = H_i.inverse() * f_i;
                x_new[i] = cur_positions[i] + delta_xi;

                bool dog = x_new[i].hasNaN();
            }
            else {
                x_new[i] = cur_positions[i];
            }
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
bool first = true;
void Mesh::initialGuess(float dt, const Eigen::Vector3f& a_ext) {
    size_t num_vertices = init_positions.size();

    if (initGuessType == initGuessEnum::ADAPTIVE) {
        float a_ext_mag = a_ext.norm();
        Eigen::Vector3f a_ext_hat = Eigen::Vector3f::Zero();

        // Only normalize if there's a meaningful external acceleration
        if (a_ext_mag > FLOAT_EPS) {
            a_ext_hat = a_ext / a_ext_mag;
        }

        #pragma omp parallel for
        for (size_t i = 0; i < num_vertices; i++) {
            // Compute the approximate previous frame acceleration
            Eigen::Vector3f a_t = (cur_velocities[i] - prev_velocities[i]) / dt;

            // Project this acceleration onto the direction of a_ext
            float a_t_ext = 0.0f;
            if (a_ext_mag > FLOAT_EPS) {
                a_t_ext = a_t.dot(a_ext_hat);
            }

            // Determine the scaling coefficient
            float a_coef;
            if (a_t_ext > a_ext_mag) {
                a_coef = 1.0f;
            }
            else if (a_t_ext < 0.0f) {
                a_coef = 0.0f;
            }
            else {
                a_coef = (a_ext_mag > FLOAT_EPS) ? (a_t_ext / a_ext_mag) : 0.0f;
            }

            Eigen::Vector3f a_tilde = a_coef * a_ext;
            if (first) {
                a_tilde = a_ext;
                first = false;
            }

            // y = x_t + dt * v_t + dt^2 * a_ext no matter what
            y[i] = prev_positions[i] + (dt * prev_velocities[i]) + (dt * dt * a_ext);
            cur_positions[i] = prev_positions[i] + (dt * prev_velocities[i]) + (dt * dt * a_tilde);
        }
    }
    else if (initGuessType == initGuessEnum::INERTIA_ACCEL) {
        #pragma omp parallel for
        for (size_t i = 0; i < num_vertices; i++) {
            // y = x = x_t + hv_t + h^2a_ext
            y[i] = prev_positions[i] + (dt * prev_velocities[i]) + (dt * dt * a_ext);
            prev_positions[i] = cur_positions[i];
            cur_positions[i] = y[i];
        }
    }
}

void Mesh::updateVelocities(float dt) {
    float inv_dt = 1.0f / dt;

    const float max_vMag = 10.0f;
    const float max_vMagSq = max_vMag * max_vMag;

    #pragma omp parallel for
    for (size_t i = 0; i < cur_velocities.size(); i++) {
        prev_velocities[i] = cur_velocities[i];
        cur_velocities[i] = (cur_positions[i] - prev_positions[i]) * inv_dt;
        auto tmp = prev_positions[i];
        prev_positions[i] = cur_positions[i];

        // Damping
        float v_mag = cur_velocities[i].norm();
        if (v_mag < FLOAT_EPS) {
            continue;
        }

        bool dog = cur_velocities[i].hasNaN();

        // if (cur_velocities[i].hasNaN()) {
        //     cout << "i = " << i << " with NaN, v_mag: " << v_mag << endl;
        //     printvec3(cur_velocities[i]);
        //     printvec3(cur_positions[i]);
        //     printvec3(tmp);
        // }

        cur_velocities[i] *= (max_vMag / v_mag);
    }
}

void Mesh::writeToVTK(const string& output_dir, bool raw) {
    vtkSmartPointer<vtkPoints> vtk_points = vtkSmartPointer<vtkPoints>::New();
    vtk_points->SetDataTypeToFloat();

    for (const auto& pos : prev_positions) {
        vtk_points->InsertNextPoint(pos.x(), pos.y(), pos.z());
    }

    vtkSmartPointer<vtkCellArray> vtk_cells = vtkSmartPointer<vtkCellArray>::New();
    vtk_cells->Allocate(tetrahedra.size());
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

void Mesh::initFromJson(const json& mesh_data) {
    // Extract material and physical parameters from JSON
    mass    = mesh_data["params"]["mass"].get<float>();
    mu      = mesh_data["params"]["mu"].get<float>();
    lambda  = mesh_data["params"]["lambda"].get<float>();
    damping = mesh_data["params"]["damping"].get<float>();
    k_c     = mesh_data["params"]["k_c"].get<float>();
    mu_c    = mesh_data["params"]["mu_c"].get<float>();
    eps_c   = mesh_data["params"]["eps_c"].get<float>();

    scale = Eigen::Vector3f(
        mesh_data["params"]["scale"][0].get<float>(),
        mesh_data["params"]["scale"][1].get<float>(),
        mesh_data["params"]["scale"][2].get<float>()
    );

    position = Eigen::Vector3f(
        mesh_data["params"]["position"][0].get<float>(),
        mesh_data["params"]["position"][1].get<float>(),
        mesh_data["params"]["position"][2].get<float>()
    );

    velocity = Eigen::Vector3f(
        mesh_data["params"]["velocity"][0].get<float>(),
        mesh_data["params"]["velocity"][1].get<float>(),
        mesh_data["params"]["velocity"][2].get<float>()
    );

    // Assume initial guess type is always inertia + accel
    initGuessType = initGuessEnum::INERTIA_ACCEL;
    initGuessType = initGuessEnum::ADAPTIVE;

    std::cout << "Parameters:" << std::endl;
    std::cout << "\tmass: "    << mass    << std::endl;
    std::cout << "\tmu: "      << mu      << std::endl;
    std::cout << "\tlambda: "  << lambda  << std::endl;
    std::cout << "\tdamping: " << damping << std::endl;
    std::cout << "\tk_c: "     << k_c     << std::endl;
    std::cout << "\tmu_c: "    << mu_c    << std::endl;
    std::cout << "\teps_c: "   << eps_c   << std::endl;
    std::cout << "\tposition: " << position.x() << " " << position.y() << " " << position.z() << std::endl;
    std::cout << "\tvelocity: " << velocity.x() << " " << velocity.y() << " " << velocity.z() << std::endl;
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
        Eigen::Vector3f pos(
            static_cast<float>(vertex[0]),
            static_cast<float>(vertex[1]),
            static_cast<float>(vertex[2])
        );

        pos += position;

        // scale
        pos = pos.cwiseProduct(scale);

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
            array<int, 4> tet_idx;
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
    std::vector<std::vector<int>> color_groups;
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
    vector<Eigen::Vector3f> new_positions, new_velocities;
    vector<int> new_colors;
    vector<int> old2new(num_vertices);

    new_positions.reserve(num_vertices);
    new_velocities.reserve(num_vertices);
    new_colors.reserve(num_vertices);

    size_t idx = 0;
    cout << "Sorting vertices by color to improve processing efficiency" << endl;
    // for (int color = 0; color < color_groups.size(); color++) {
    //     for (int i = 0; i < color_groups[color].size(); i++) {
    //         int old_idx = color_groups[color][i];

    //         old2new[old_idx] = idx;

    //         new_positions.push_back(init_positions[old_idx]);
    //         new_velocities.push_back(cur_velocities[old_idx]);
    //         new_colors.push_back(colors[old_idx]);

    //         idx++;
    //     }
    // }

    // After sort, check old tetrahedra[0] (after next step we will check again with old2new)
    const auto& old_tet = tetrahedra[0];
    cout << "Old tetrahedra[0] = [" << old_tet[0] << ", " << old_tet[1] << ", " << old_tet[2] << ", " << old_tet[3] << "]" << endl;

    // Each index stored in a given tetrahedra is old, so we should update to the new one
    // for (auto& tet : tetrahedra) {
    //     for (auto& v_idx : tet) {
    //         v_idx = old2new[v_idx];
    //     }
    // }

    // We need to store one way access to all tetrahedra indices given a vertex for quick access to neighbors
    vertex2tets.resize(num_vertices);
    for (int i = 0; i < tetrahedra.size(); i++) {
        for (int j = 0; j < 4; j++) {
            vertex2tets[tetrahedra[i][j]].push_back(i);
        }
    }

    const auto& new_tet = tetrahedra[old2new[0]];
    cout << "New tetrahedra[0] = [" << new_tet[0] << ", " << new_tet[1] << ", " << new_tet[2] << ", " << new_tet[3] << "]" << endl;

    // Update prev_pos, cur_pos, and y positions
    // prev_positions = new_positions;
    // cur_positions = new_positions;
    // y = new_positions;
    prev_positions = init_positions;
    cur_positions = init_positions;
    y = init_positions;

    // Swap the old arrays for the new ones
    // init_positions = std::move(new_positions);
    // cur_velocities = std::move(new_velocities);
    // colors = std::move(new_colors);

    // Precalculate relevant tetrahedron data
    Dm_inverses.resize(tetrahedra.size());
    tet_volumes.resize(tetrahedra.size());
    for (int i = 0; i < tetrahedra.size(); i++) {
        const array<int, 4>& tet = tetrahedra[i];
        
        Eigen::Vector3f x0 = init_positions[tet[0]];
        Eigen::Vector3f x1 = init_positions[tet[1]];
        Eigen::Vector3f x2 = init_positions[tet[2]];
        Eigen::Vector3f x3 = init_positions[tet[3]];

        // Dm is the reference shape matrix, which is defined by the three edge vectors
        Eigen::Matrix3f Dm;
        Dm.col(0) = x1 - x0;
        Dm.col(1) = x2 - x0;
        Dm.col(2) = x3 - x0;

        Dm_inverses[i] = Dm.inverse();
        float vol = fabs(Dm.determinant() / 6.0f);
        if (fabs(vol) < FLOAT_EPS) {
            cout << "Degenerate tetrahedron " << i << " has zero volume = " << vol << endl;
            printvec3(x0);
            printvec3(x1);
            printvec3(x2);
            printvec3(x3);
        }
        tet_volumes[i] = vol;
    }

    // For fun we can evaluate density
    vector<float> color_freq(color_groups.size());
    for (int i = 0; i < color_groups.size(); i++) {
        color_freq[i] = (float)color_groups[i].size();
    }

    std::cout << "Theoretical even vertices per group (not necessarily ideal) = " 
              << ((float)num_vertices / color_groups.size()) 
              << " versus actual: ";
    for (const auto& freq : color_freq) {
        std::cout << freq << " ";
    }
    std::cout << std::endl;

    // Calculate color ranges. This is coming along so nicely!
    size_t start_idx = 0;
    color_ranges.reserve(color_groups.size());
    for (int color = 0; color < color_groups.size(); color++) {
        size_t group_sz = color_groups[color].size();

        // Apparently emplace back is better for on the fly construction push back
        color_ranges.emplace_back(std::array<size_t, 2>{start_idx, start_idx + group_sz});
        start_idx += group_sz;
    }

    // Print color ranges
    for (size_t i = 0; i < color_ranges.size(); i++) {
        std::cout << "\tColor " << i << ": [" << color_ranges[i][0] 
                  << ", " << color_ranges[i][1] << ")" << std::endl;
    }

    // Sanity check
    assert(init_positions.size()  == num_vertices);
    assert(prev_velocities.size() == num_vertices);
    assert(colors.size()          == num_vertices);
}