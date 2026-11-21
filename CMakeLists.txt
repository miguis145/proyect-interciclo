cmake_minimum_required(VERSION 3.16)
project(proyecto_interciclo LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)   # el benchmark CPU vs GPU solo tiene sentido con optimizaciones
endif()

# Si tu OpenCV está en una ruta propia:  cmake -DOpenCV_DIR=/ruta/lib/cmake/opencv4 ..
find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs videoio highgui)

add_executable(vision.bin ProyectoVision.cpp)
target_include_directories(vision.bin PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(vision.bin PRIVATE ${OpenCV_LIBS})
