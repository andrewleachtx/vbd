cd "C:/Users/andre/Desktop/code/graphics/vbd"

Remove-Item -Recurse -Force .\build\

cmake -B build/ -DCMAKE_BUILD_TYPE=Debug -DCMAKE_TOOLCHAIN_FILE="C:/Dev/vcpkg/scripts/buildsystems/vcpkg.cmake"

cmake --build build/ --config Debug --parallel