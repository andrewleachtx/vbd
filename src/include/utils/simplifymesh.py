import pymeshlab
import sys

"""
    Decimates polygons in a mesh with the hope of reducing tetrahedra

    https://www.ijraset.com/best-journal/mesh-optimization-using-python-libraries#:~:text=Mesh%20decimation%20is%20a%20technique,mesh%20simplification%20or%20mesh%20reduction.
"""

if len(sys.argv) < 3:
    print("Usage: python simplify.py <input_file> <percentage> [output_file]")
    sys.exit(1)

INPUT_OBJ = sys.argv[1]
if INPUT_OBJ[-4:] != ".obj":
    print(f"Please use a proper obj file for the desired_obj, you inputted {INPUT_OBJ}")
    sys.exit(1)

REDUCTION_PERCENT = float(sys.argv[2])
if REDUCTION_PERCENT < 0 or REDUCTION_PERCENT > 1:
    print(f"Please use a goal percentage of the original mesh size between 0 and 1, you inputted {REDUCTION_PERCENT}")
    sys.exit(1)

RESOURCE_DIR = "./resources/models"
if len(sys.argv) == 4:
    RESOURCE_DIR = sys.argv[2]

if RESOURCE_DIR[-1] == "/":
    RESOURCE_DIR = RESOURCE_DIR[:-1]

INPUT_FILE = f"{RESOURCE_DIR}/obj/{INPUT_OBJ}"
OUTPUT_FILE = f"{RESOURCE_DIR}/obj/{INPUT_OBJ.replace('.obj', '_simplified.obj')}"

ms = pymeshlab.MeshSet()

print(f"Loading mesh from {INPUT_FILE}")
ms.load_new_mesh(INPUT_FILE)

# https://pymeshlab.readthedocs.io/en/latest/filter_list.html#meshing_decimation_quadric_edge_collapse
ms.apply_filter("meshing_decimation_quadric_edge_collapse", targetperc=REDUCTION_PERCENT, preservenormal=True)

print(f"Saving simplified mesh to {OUTPUT_FILE}")
ms.save_current_mesh(OUTPUT_FILE)
