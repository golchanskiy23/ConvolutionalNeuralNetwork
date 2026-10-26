cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(server)

# Путь к LibTorch передаётся при конфигурации, например:
#   cmake -S . -B build -DCMAKE_PREFIX_PATH=/opt/libtorch
#   cmake -S . -B build -DCMAKE_PREFIX_PATH="$(python -c 'import torch; print(torch.utils.cmake_prefix_path)')"
find_package(Torch REQUIRED)
find_package(Threads REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(server main.cpp
        ImageClassification.h
        ImageClassification.cpp
        WebServer.cpp)
target_include_directories(server PRIVATE third_party)

target_link_libraries(server
        ${TORCH_LIBRARIES}
        Threads::Threads
)
# LibTorch 2.x требует C++20
set_property(TARGET server PROPERTY CXX_STANDARD 20)
