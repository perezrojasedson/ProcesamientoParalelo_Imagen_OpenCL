cmake_minimum_required(VERSION 3.20)
project(taller_opencl C)

set(CMAKE_C_STANDARD 99)

# Buscar el paquete de OpenCL en tu sistema
find_package(OpenCL REQUIRED)

# Incluir el directorio actual para encontrar los .h de stb
include_directories(.)

add_executable(taller_opencl main.c)

# Vincular el ejecutable con la librería OpenCL encontrada
target_link_libraries(taller_opencl OpenCL::OpenCL)