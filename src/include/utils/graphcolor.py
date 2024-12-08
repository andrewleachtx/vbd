import time
import meshio
import networkx as nx
import numpy as np
import sys

"""
    This code takes in a .vtk (which can be generated with tetrahedralize.py) and will use
    a greedy coloring algorithm to color vertices, and write to a new .vtk file with color
    data in the mesh.point_data section
"""

### SCRIPT/ARGS HANDLING ###
if len(sys.argv) < 2:
    print("Usage: python graphcolor.py <desired_vtk> [resource_dir]")

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
OUTPUT_FILE = f"{RESOURCE_DIR}/vtk/{INPUT_VTK.replace('.vtk', '_c.vtk')}"

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
time_start = time.time()
for tet in cells:
    for i in range(4):
        for j in range(i + 1, 4):
            G.add_edge(tet[i], tet[j])
time_stop = time.time()
print(f"Edges added in {(time_stop - time_start):.6f}s with {G.number_of_nodes()} vertices and {G.number_of_edges()} edges")

# Actually do the graph coloring, I will use greedy
time_start = time.time()
colored_graph = nx.coloring.greedy_color(G, strategy="largest_first")
time_stop = time.time()
unique_colors = set(colored_graph.values())
num_colors = len(unique_colors)
print(f"Graph coloring done in {(time_stop - time_start):.6f}s with {num_colors} colors")

# We can now rewrite the mesh with colors added as "point_data"
color_data = np.zeros((len(points), 1))
for vertex, color in colored_graph.items():
    color_data[vertex] = color

### VALID CHECK ###
def is_valid(G, colors):
    for u, v in G.edges():
        if colors[u] == colors[v]:
            return False
    return True

if not is_valid(G, colored_graph):
    print("Graph coloring was invalid")
    sys.exit(1)

### WRITE TO NEW VTK ###
mesh.point_data = {"color": color_data}
try:
    meshio.write(OUTPUT_FILE, mesh)
    print(f"Successfuly colored {INPUT_VTK} and wrote to {OUTPUT_FILE}")
except:
    print(f"Failed to write to {OUTPUT_FILE}")
    sys.exit(1)

