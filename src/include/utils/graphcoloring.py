import meshio
import networkx as nx
import numpy as np
import sys

"""
    This code takes in a .vtk (which can be generated with tetrahedralize.py) and will use
    a greedy coloring algorithm to color vertices.
"""


### SCRIPT/ARGS HANDLING ###
if len(sys.argv) < 2:
    print("Usage: python graphcoloring.py <desired_vtk> [resource_dir]")

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
OUTPUT_FILE = f"{RESOURCE_DIR}/vtk/{INPUT_VTK.replace('.vtk', '_color.vtk')}"

### MESH LOADING & GRAPH ALGO ###
mesh = meshio.read(INPUT_FILE)

# Grab points from the vtk
points = mesh.points
cells = mesh.cells_dict.get("tetra", [])

if len(cells) == 0:
    print(f"No tetrahedra were found in {INPUT_VTK}")
    sys.exit(1)

# Construct the graph and add a vertex identified by each point
G = nx.Graph()
G.add_nodes_from(range(len(points)))

# Create edges between tetrahedron vertex a_i, b_i, c_i, d_i
for tet in cells:
    for i in range(4):
        for j in range(i + 1, 4):
            G.add_edge(tet[i], tet[j])

# Actually do the graph coloring, I will use greedy
colored_graph = nx.coloring.greedy_color(G, strategy="largest_first")
unique_colors = set(colored_graph.values())
print(f"Colored graph of {G.number_of_nodes()} vertices and {G.number_of_edges()} edges with {len(unique_colors)} colors!")