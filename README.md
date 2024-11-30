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
- nlohmann-json
- 


I use vcpkg, but you can link packages however works for you - while in the project source directory, run
`cmake -B build/ -DCMAKE_BUILD_TYPE=Release -DCMAKE_TOOLCHAIN_FILE="C:/Dev/vcpkg/scripts/buildsystems/vcpkg.cmake"`

Then, `cmake --build build/ --config Release --parallel`
