cmake --build build/ --config Release --parallel

if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

# cmake --build build/ --config Debug --parallel

# ./build/Release/VBD.exe <resource_dir> <scene_no> <state_output_dir> <is_usingGPU = true | false>
./build/Release/VBD.exe ./resources 0 ./output true
# 