import tetgen
import meshio
import numpy as np
import sys

"""
    Mesh tetrahedralizer with tetgen and meshio

    If you get errors in the tetrahedralization itself, you may need to open the mesh
    in MeshLab and run:
        - Filters > Cleaning and Reparing
            - Remove duplicated faces, vertices & remove isolated pieces
        - Filters > Filters > Remeshing, Simplification, and Reconstruction
            - Surface Reconstruction: Screened Poisson
"""

if len(sys.argv) < 2:
    print("Usage: python tetrahedralize.py <desired_obj> [resource_dir]")
    sys.exit(1)

desired_obj = sys.argv[1]
if desired_obj[-4:] != ".obj":
    print(f"Please use a proper obj file for the desired_obj, you inputted {desired_obj}")
    sys.exit(1)

resource_dir = "./resources/models"
if len(sys.argv) == 3:
    resource_dir = sys.argv[2]

if resource_dir[-1] != "/":
    resource_dir += "/"

input_file = f"{resource_dir}{desired_obj}"
mesh = meshio.read(input_file)

tgen = tetgen.TetGen(mesh.points, mesh.cells_dict["triangle"])
tgen.tetrahedralize(order=1, mindihedral=20, minratio=1.5)

tet_points = tgen.node
tet_elements = tgen.elem

tetra_mesh = meshio.Mesh(
    points=tet_points,
    cells=[("tetra", tet_elements)]
)

output_file = f"{resource_dir}{desired_obj.replace('.obj', '.vtk')}"
meshio.write(output_file, tetra_mesh)

print(f"Successfully tetrahedralized {desired_obj} and stored in {output_file}")
