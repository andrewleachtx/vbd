cmake --build build/ --config Release --parallel
# cmake --build build/ --config Debug --parallel

# ./build/Release/VBD.exe <resource_dir> <scene_no> <state_output_dir>
./build/Release/VBD.exe ./resources 0 ./output
# ./build/Debug/VBD.exe ./resources 0 ./output
