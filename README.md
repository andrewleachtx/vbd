# Vertex Block Descent

## Andrew Leach
["Everything should be made as simple as possible, but not simpler."](https://matthias-research.github.io/pages/index.html)

### Sources & References
- Vertex Block Descent [paper](https://doi.org/10.1145/3658179).
- Anka Chen's [Gaia](https://github.com/AnkaChan/Gaia) with VBD implemented.

### Everything Happens in `src/main.cpp`

### Building
Dependencies used are
- CMake
- C++17
- CUDA 11.8
- nlohmann-json (reading scene data json in cpp)
- meshio (reading .obj & .vtk)
- tetgen (for tetrahedralization)
- networkx (for greedy graph coloring)
- vtk

I use vcpkg, but you can link packages however works for you - while in the project source directory, you can either run `./rebuild.ps1` or across environments that just runs
`cmake -B build/ -DCMAKE_BUILD_TYPE=Release -DCMAKE_TOOLCHAIN_FILE="C:/Dev/vcpkg/scripts/buildsystems/vcpkg.cmake"`

Then, `cmake --build build/ --config Release --parallel`

### Adding New Models
Start by loading the venv with 

```
python -m venv ./env
.\env\Scripts\activate # (on bash-like shells use source ./env/bin/activate on bash)
pip install -r requirements.txt
```

The full pipeline starts by adding a `.obj` into `resources/models`, where you can then

1. (OPTIONAL) Simplify mesh, reducing vertice count with `python simplify.py <desired_obj> <percent> [resource_dir]` which outputs `models/obj/<desired_objname>_simplified.obj`
2. Tetrahedralize any obj with `python tetrahedralize.py <desired_obj> [resource_dir]`.
3. Graph color the tetrahedralized `.vtk` produced in 2 with `python graphcolor.py <desired_vtk> [resource_dir]`, creating `models/vtk/<desired_vtkname>_colored.vtk`.
4. From here you can add scenes to `scene.json`, and change main.cpp based on the desired scene number. This is not fully tested.