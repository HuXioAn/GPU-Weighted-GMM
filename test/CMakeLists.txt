cmake_minimum_required(VERSION 3.21)
project(GMMTest LANGUAGES CXX)

include(CTest)
enable_testing()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

cmake_policy(SET CMP0042 NEW)
set(CMAKE_BUILD_RPATH_USE_ORIGIN TRUE)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

# Language must be enabled here (top-level project) before add_subdirectory.
option(HIP_ON "Use HIP backend instead of CUDA" OFF)
if(HIP_ON)
	find_package(HIP REQUIRED)
	enable_language(HIP)
else()
	enable_language(CUDA)
	set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo")
endif()

set(GPUWeightedGMM_DIR
	"${CMAKE_CURRENT_LIST_DIR}/.."
	CACHE PATH
	"Path to the GPU-Weighted-GMM source tree")
get_filename_component(GPUWeightedGMM_DIR "${GPUWeightedGMM_DIR}" ABSOLUTE)

find_package(GPUWeightedGMM REQUIRED)

# ---------------------------------------------------------------------------
# Helper: create a GPU test executable linked against gpu_weighted_gmm.
# Includes, HIPIFLY define, HIP link/compile options all propagate from the
# interface target; only CUDA-specific LTO suppression stays here.
# ---------------------------------------------------------------------------
macro(add_gmm_test target source)
	add_executable(${target} ${source})
	if(HIP_ON)
		set_source_files_properties(${source} PROPERTIES LANGUAGE HIP)
	else()
		set_source_files_properties(${source} PROPERTIES LANGUAGE CUDA)
		set_target_properties(${target} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
		set_property(TARGET ${target} PROPERTY INTERPROCEDURAL_OPTIMIZATION FALSE)
		set_property(TARGET ${target} PROPERTY LINK_OPTIONS -fno-lto)
	endif()
	target_link_libraries(${target} PRIVATE gpu_weighted_gmm)
	add_test(NAME ${target} COMMAND ${target})
	set_tests_properties(${target} PROPERTIES TIMEOUT 10)
endmacro()

add_gmm_test(testHistogram3D  testHistogram3D.cu)
add_gmm_test(testHistogram2D  testHistogram2D.cu)
add_gmm_test(testGMM          testGMM.cu)
add_gmm_test(testCompressor2D testCompressor2D.cu)
add_gmm_test(testCompressor3D testCompressor3D.cu)




