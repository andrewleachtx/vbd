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

INPUT_OBJ = sys.argv[1]
if INPUT_OBJ[-4:] != ".obj":
    print(f"Please use a proper obj file for the desired_obj, you inputted {INPUT_OBJ}")
    sys.exit(1)

RESOURCE_DIR = "./resources/models"
if len(sys.argv) == 3:
    RESOURCE_DIR = sys.argv[2]

if RESOURCE_DIR[-1] == "/":
    RESOURCE_DIR = RESOURCE_DIR[:-1]

INPUT_FILE = f"{RESOURCE_DIR}/obj/{INPUT_OBJ}"
OUTPUT_FILE = f"{RESOURCE_DIR}/vtk/{INPUT_OBJ.replace('.obj', '.vtk')}"
mesh = meshio.read(INPUT_FILE)

tgen = tetgen.TetGen(mesh.points, mesh.cells_dict["triangle"])
tgen.tetrahedralize(order=1, mindihedral=20, minratio=1.5, verbose=True)

tet_points = tgen.node
tet_elements = tgen.elem

tetra_mesh = meshio.Mesh(
    points=tet_points,
    cells=[("tetra", tet_elements)]
)

meshio.write(OUTPUT_FILE, tetra_mesh)

print(f"Successfully tetrahedralized {INPUT_OBJ} with {len(tet_elements)} tetrahedra and stored in {OUTPUT_FILE}")