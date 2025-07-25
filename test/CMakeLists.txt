cmake_minimum_required(VERSION 3.21) # HIP requires 3.21
project(GMMTest LANGUAGES CXX)

include(CTest)
enable_testing()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

cmake_policy(SET CMP0042 NEW)
set(CMAKE_BUILD_RPATH_USE_ORIGIN TRUE)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)


option(HIP_ON "Use HIP" OFF) # Set it to OFF if you want to use CUDA


if(NOT HIP_ON) # CUDA
    enable_language(CUDA)
    message(STATUS "Using CUDA environment.")
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)

    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo")

elseif(HIP_ON) # HIP
    find_package(HIP REQUIRED)
    enable_language(HIP)
    message(STATUS "Using HIP environment.")
    set(DEVICE_LANGUAGE HIP)
    set(DEVICE_COMPILER hipcc)

endif()


set(common_include_path ${CMAKE_CURRENT_SOURCE_DIR}/../common)

if(NOT HIP_ON)
  list(APPEND common_include_path ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
else()
  list(APPEND common_include_path ${CMAKE_CURRENT_SOURCE_DIR}/../hipifly)
endif()


# testHistogram 3D

set(test_histogram_root ${CMAKE_CURRENT_SOURCE_DIR}/../histogram)

file(
  GLOB
  testHistogram3D_SRC
  ${test_histogram_root}/*.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/testHistogram3D.cpp
)

add_executable(testHistogram3D ${testHistogram3D_SRC})
target_include_directories(testHistogram3D PRIVATE ${common_include_path} ${test_histogram_root})
if(HIP_ON)
    target_compile_definitions(testHistogram3D PUBLIC HIPIFLY)
endif()

if(NOT HIP_ON) # CUDA

  set_target_properties(testHistogram3D PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
  set_property(TARGET testHistogram3D PROPERTY CUDA_ARCHITECTURES 75) # required for double atomicadd
  set_property(TARGET testHistogram3D PROPERTY CUDA_STANDARD 17)
  set_property(TARGET testHistogram3D PROPERTY INTERPROCEDURAL_OPTIMIZATION FALSE)
  set_property(TARGET testHistogram3D PROPERTY LINK_OPTIONS -fno-lto)

else() # HIP

  target_link_libraries(testHistogram3D PRIVATE hip::device)
  target_link_libraries(testHistogram3D PRIVATE hip::host)

  target_compile_options(testHistogram3D PUBLIC -fgpu-rdc)
  target_link_options(testHistogram3D PUBLIC  -fgpu-rdc --hip-link)

  set_property(TARGET testHistogram3D PROPERTY HIP_ARCHITECTURES gfx90a)

endif()


add_test(
  NAME testHistogram3D
  COMMAND testHistogram3D
)
set_tests_properties(testHistogram3D PROPERTIES TIMEOUT 10)



# testHistogram 2D

set(test_histogram_root ${CMAKE_CURRENT_SOURCE_DIR}/../histogram)

file(
  GLOB
  testHistogram2D_SRC
  ${test_histogram_root}/*.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/testHistogram2D.cpp
)

add_executable(testHistogram2D ${testHistogram2D_SRC})
target_include_directories(testHistogram2D PRIVATE ${common_include_path} ${test_histogram_root})
if(HIP_ON)
    target_compile_definitions(testHistogram2D PUBLIC HIPIFLY)
endif()

if(NOT HIP_ON) # CUDA

  set_target_properties(testHistogram2D PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
  set_property(TARGET testHistogram2D PROPERTY CUDA_ARCHITECTURES 75) # required for double atomicadd
  set_property(TARGET testHistogram2D PROPERTY CUDA_STANDARD 17)
  set_property(TARGET testHistogram2D PROPERTY INTERPROCEDURAL_OPTIMIZATION FALSE)
  set_property(TARGET testHistogram2D PROPERTY LINK_OPTIONS -fno-lto)

else() # HIP

  target_link_libraries(testHistogram2D PRIVATE hip::device)
  target_link_libraries(testHistogram2D PRIVATE hip::host)

  target_compile_options(testHistogram2D PUBLIC -fgpu-rdc)
  target_link_options(testHistogram2D PUBLIC  -fgpu-rdc --hip-link)

  set_property(TARGET testHistogram2D PROPERTY HIP_ARCHITECTURES gfx90a)

endif()


add_test(
  NAME testHistogram2D
  COMMAND testHistogram2D
)
set_tests_properties(testHistogram2D PROPERTIES TIMEOUT 10)


# testGMM

set(test_gmm_root ${CMAKE_CURRENT_SOURCE_DIR}/../weighted-GMM)

file(
  GLOB
  testGMM_SRC
  ${CMAKE_CURRENT_SOURCE_DIR}/testGMM.cu
)

add_executable(testGMM ${testGMM_SRC})
target_include_directories(testGMM PRIVATE ${common_include_path} ${test_gmm_root})
if(HIP_ON)
    target_compile_definitions(testGMM PUBLIC HIPIFLY)
endif()


if(NOT HIP_ON) # CUDA

  set_target_properties(testGMM PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
  set_property(TARGET testGMM PROPERTY CUDA_ARCHITECTURES 75) # required for double atomicadd
  set_property(TARGET testGMM PROPERTY CUDA_STANDARD 17)
  set_property(TARGET testGMM PROPERTY INTERPROCEDURAL_OPTIMIZATION FALSE)
  set_property(TARGET testGMM PROPERTY LINK_OPTIONS -fno-lto)

else() # HIP

  target_link_libraries(testGMM PRIVATE hip::device)
  target_link_libraries(testGMM PRIVATE hip::host)

  target_compile_options(testGMM PUBLIC -fgpu-rdc)
  target_link_options(testGMM PUBLIC  -fgpu-rdc --hip-link)

  set_property(TARGET testGMM PROPERTY HIP_ARCHITECTURES gfx90a)

endif()


add_test(
  NAME testGMM
  COMMAND testGMM
)
set_tests_properties(testGMM PROPERTIES TIMEOUT 10)


# # testPipeline

# add_executable(testPipeline ${testHistogram3D_SRC} ${testGMM_SRC} ${CMAKE_CURRENT_SOURCE_DIR}/testPipeline.cpp)
# target_include_directories(testPipeline PRIVATE ${common_include_path} ${test_histogram_root} ${test_gmm_root})
# if(HIP_ON)
#     target_compile_definitions(testPipeline PUBLIC HIPIFLY)
# endif()

# add_test(
#   NAME testPipeline
#   COMMAND testPipeline
# )
# set_tests_properties(testPipeline PROPERTIES TIMEOUT 10)




