cmake_minimum_required(VERSION 3.1 FATAL_ERROR)
project(pytorch3d_operators LANGUAGES CXX CUDA)

include(${CMAKE_BINARY_DIR}/conan_paths.cmake)

find_package(Torch REQUIRED)
find_package(CUDA REQUIRED)

# Define our library target
set(HEADERS
init.h
)

set(SOURCES
init.cpp
knn.cpp
)

set(CUDA_SOURCES
dispatch.cuh
index_utils.cuh
knn.cu
mink.cuh
)

add_library(pytorch3d_operators SHARED ${HEADERS} ${SOURCES} ${CUDA_SOURCES})

# Custom flags
if(MSVC)
    set(CMAKE_CUDA_FLAGS "-Xcompiler=\"/EHsc /wd4267 /wd4251 /wd4522 /wd4838 /wd4305 /wd4244 /wd4190 /wd4101 /wd4996 /wd4275 /bigobj\"")
endif()

# Enable C++17
target_compile_features(pytorch3d_operators PRIVATE cxx_range_for)
target_compile_features(pytorch3d_operators PUBLIC cxx_std_17)

# Link against LibTorch
target_link_libraries(pytorch3d_operators torch cuda)

if(${CMAKE_VERSION} GREATER_EQUAL "3.18")
    set_property(TARGET pytorch3d_operators PROPERTY CUDA_ARCHITECTURES 50 52 60 61 70 75 80 86)
endif()

if (MSVC)
    foreach( OUTPUTCONFIG ${CMAKE_CONFIGURATION_TYPES} )
        install(TARGETS pytorch3d_operators EXPORT pytorch3d_operators
            ARCHIVE DESTINATION lib/${OUTPUTCONFIG}
            LIBRARY DESTINATION lib/${OUTPUTCONFIG}
            RUNTIME DESTINATION bin/${OUTPUTCONFIG})

        install(FILES $<TARGET_PDB_FILE:pytorch3d_operators> DESTINATION lib/${OUTPUTCONFIG} OPTIONAL)

        install(EXPORT pytorch3d_operators DESTINATION lib/cmake/pytorch3d_operators FILE pytorch3d_operators-config.cmake)
    endforeach( OUTPUTCONFIG CMAKE_CONFIGURATION_TYPES )
else()
    install(TARGETS pytorch3d_operators EXPORT pytorch3d_operators
        ARCHIVE DESTINATION lib/pytorch3d_operators
        LIBRARY DESTINATION lib/pytorch3d_operators
        RUNTIME DESTINATION bin/pytorch3d_operators)
    install(EXPORT pytorch3d_operators DESTINATION lib/cmake/pytorch3d_operators FILE pytorch3d_operators-config.cmake)
endif(MSVC)
