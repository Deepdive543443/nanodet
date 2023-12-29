set(name_proj Nanotest)

cmake_minimum_required(VERSION 3.0.0)
project(${name_proj} VERSION 0.1.0 LANGUAGES C CXX)

add_executable(${name_proj} main.cpp)

set(OpenCV_DIR ../../../../cpp_libs/opencv-mobile-2.4.13.7-ubuntu-2204/share/OpenCV)
# target_link_libraries(${name_proj} curlpp)

# set(OpenCV_DIR ../../../../cpp_libs/opencv-4.8.0/build)
find_package(OpenCV REQUIRED)
# include_directories( ${OpenCV_INCLUDE_DIRS} )
target_link_libraries(${name_proj} ${OpenCV_LIBS} )

set(ncnn_DIR ../../../../cpp_libs/ncnn_x86/lib/cmake/ncnn CACHE PATH "Directory that contains ncnnConfig.cmake")
# set(ncnn_DIR ../../cpp_libs/ncnn/x86/lib/cmake/ncnn CACHE PATH "Directory that contains ncnnConfig.cmake")
find_package(ncnn REQUIRED)
target_link_libraries(${name_proj} ncnn)


find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()