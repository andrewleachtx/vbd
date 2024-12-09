Remove-Item -Recurse -Force .\build\

cmake -B build/ -DCMAKE_BUILD_TYPE=Release -DCMAKE_TOOLCHAIN_FILE="C:/Dev/vcpkg/scripts/buildsystems/vcpkg.cmake"
# cmake -B build/ -DCMAKE_BUILD_TYPE=Release
# cmake -B build/ -G "Visual Studio 17 2022" -T v142 -A x64 -DCMAKE_BUILD_TYPE=Release -DCMAKE_TOOLCHAIN_FILE="C:/Dev/vcpkg/scripts/buildsystems/vcpkg.cmake" -DCUDA_TOOLKIT_ROOT_DIR="C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.8/" --trace

cmake --build build/ --config Release --verbose