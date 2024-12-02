import polyscope as ps
import polyscope.imgui as psim

import meshio
import numpy as np
import matplotlib.pyplot as plt
import sys

# https://polyscope.run/py/structures/volume_mesh/basics/
# https://polyscope.run/py/structures/volume_mesh/color_quantities/
# https://matplotlib.org/stable/users/explain/colors/colormaps.html

if len(sys.argv) < 2:
    print("Usage: python visualize.py <desired_vtk> [resource_dir]")

    sys.exit(1)

INPUT_VTK = sys.argv[1]
if INPUT_VTK[-4:] != ".vtk":
    print(f"Please use a proper vtk file for the desired_vtk, you inputted {INPUT_VTK}")
    sys.exit(1)

RESOURCE_DIR = "./resources/models"
if len(sys.argv) == 3:
    RESOURCE_DIR = sys.argv[2]

if RESOURCE_DIR[-1] == "/":
    RESOURCE_DIR = RESOURCE_DIR[:-1]

INPUT_FILE = f"{RESOURCE_DIR}/vtk/{INPUT_VTK}"

### VTK / MESH LOADING ###
ps.init()

mesh = meshio.read(INPUT_FILE)

points = mesh.points
cells = mesh.cells_dict.get("tetra", [])

if len(cells) == 0:
    print(f"No tetrahedral found in {INPUT_VTK}")
    exit(1)

# Get per-vertex color data
color_data = mesh.point_data.get("color", None)
if color_data is None:
    print(f"Color data not found in {INPUT_VTK}, did you color it beforehand?")
    exit(1)

"""
    We only care about surface faces, we can't see internal.

    This is done by hashing each face for all tetrahedra - we know if it is not on the boundary it
    will have a partner face; internal tetrahedra faces will share the same face twice, and if
    that is the case we should remove it from the "seen faces"
"""
def get_surface_faces(tetrahedra: np.ndarray) -> list[tuple[int, int, int]]:
    seen_faces = set()

    for tet in tetrahedra:
        # We can uniquely "hash" or identify a face by tuplifying the sorted vertices (so it is repeatable)
        faces = [
            tuple(sorted([tet[0], tet[1], tet[2]])),
            tuple(sorted([tet[0], tet[1], tet[3]])),
            tuple(sorted([tet[0], tet[2], tet[3]])),
            tuple(sorted([tet[1], tet[2], tet[3]])),
        ]

        for face in faces:
            if face in seen_faces:
                seen_faces.remove(face)
            else:
                seen_faces.add(face)

    return list(seen_faces)

surface_faces = np.array(get_surface_faces(cells))

# Get unique colors and sort them
color_data = np.squeeze(color_data)
unique_colors = np.unique(color_data)
num_colors = len(unique_colors)

# Map color_data to indices from 0 to num_colors - 1
color_to_index = {color: idx for idx, color in enumerate(unique_colors)}
color_indices = np.array([color_to_index[color] for color in color_data])

# Pull any matplotlib colormap, "tab20" is good and convert to RGB, max colors are 20 though
cmap = plt.get_cmap("tab20", num_colors)
rgb = cmap(color_indices)
colors = rgb[:, :3]

# Register the surface triangles as our mesh and add colors
ps_mesh = ps.register_surface_mesh("Tetrahedral Mesh", points, surface_faces)
ps_mesh.add_color_quantity("Vertex Colors", colors, defined_on='vertices')

# Rendering enhancements
# ps_mesh.set_smooth_shade(True)
ps.set_window_size(1480, 960)
def my_callbacks():
    # exit on ESC
    if psim.IsKeyPressed(psim.ImGuiKey_Escape):
        ps.unshow()

    # screenshot on s
    if psim.IsKeyPressed(psim.ImGuiKey_S):
        ps.screenshot("tetrahedral_mesh.png")

    # get window size
    if psim.IsKeyPressed(psim.ImGuiKey_W):
        print(ps.get_window_size())

ps.set_user_callback(my_callbacks)

# Show the Polyscope GUI
ps.show()