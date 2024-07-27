#@author: Steven (Jiaxun) Tang <jtang@umass.edu>
project(MLInsight)
cmake_minimum_required(VERSION 3.24)

set(CUDAToolkit_ROOT "/usr/local/cuda" CACHE STRING "CUDA toolkit root path")
find_package(CUDAToolkit REQUIRED)

if (${CUDAToolkit_FOUND})
    message(STATUS "Found CUDA. Will build MLInsight-CUDA")
else ()
    message(FATAL_ERROR "Currently, MLInsight must be compiled with CUDAToolkit.")
endif ()



# Python build definition
if(DEFINED SKBUILD)
    # prevent an unused variable warning
    set(ignoreMe "${SKBUILD}")

    # call pybind11-config to obtain the root of the cmake package (pybind11 should be install via pip).
    execute_process(COMMAND ${PYTHON_EXECUTABLE} -m pybind11 --cmakedir
            OUTPUT_VARIABLE pybind11_ROOT_RAW)
    string(STRIP ${pybind11_ROOT_RAW} pybind11_ROOT)
    find_package(pybind11 REQUIRED)

    pybind11_add_module(_mlinsight MODULE "src/MLInsight.cpp")

    target_compile_options(_mlinsight PUBLIC "-g")
    target_link_libraries(_mlinsight PUBLIC CUDA::cupti CUDA::cudart)

    # Installing the extension module to the root of the package
    install(TARGETS _mlinsight DESTINATION ".")
else()
    #Use the system pybind11. This is only for the ease of spotting compilation mistake. 
    find_package(pybind11 REQUIRED)

    pybind11_add_module(_mlinsight MODULE "src/MLInsight.cpp")
    target_link_libraries(_mlinsight PUBLIC CUDA::cupti  CUDA::cudart)
endif()
