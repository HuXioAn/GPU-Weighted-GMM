cmake_minimum_required(VERSION 3.21)
project(GPUWeightedGMM LANGUAGES CXX)

# ---------------------------------------------------------------------------
# Backend selection: CUDA (default) or HIP
# ---------------------------------------------------------------------------
option(HIP_ON "Use HIP backend instead of CUDA" OFF)

# enable_language must be called from the top-level project.
# When consumed via add_subdirectory the parent is responsible for that;
# we only do it here when this is the top-level build.
if(PROJECT_IS_TOP_LEVEL)
	if(HIP_ON)
		find_package(HIP REQUIRED)
		enable_language(HIP)
	else()
		enable_language(CUDA)
	endif()
endif()

if(HIP_ON)
	message(STATUS "GPU-Weighted-GMM: using HIP backend")
	set(CMAKE_HIP_STANDARD 17)
	set(CMAKE_HIP_STANDARD_REQUIRED ON)
	if(NOT CMAKE_HIP_ARCHITECTURES)
		set(CMAKE_HIP_ARCHITECTURES gfx90a CACHE STRING "HIP GPU architecture")
	endif()
else()
	message(STATUS "GPU-Weighted-GMM: using CUDA backend")
	set(CMAKE_CUDA_STANDARD 17)
	set(CMAKE_CUDA_STANDARD_REQUIRED ON)
	if(NOT CMAKE_CUDA_ARCHITECTURES)
		set(CMAKE_CUDA_ARCHITECTURES 75 CACHE STRING "CUDA GPU architecture")
	endif()
endif()

# ---------------------------------------------------------------------------
# Interface target
# ---------------------------------------------------------------------------
# Guard: avoid redefining the target when consumed more than once
if(TARGET gpu_weighted_gmm)
	return()
endif()

add_library(gpu_weighted_gmm INTERFACE)

target_include_directories(gpu_weighted_gmm
	INTERFACE
		"${CMAKE_CURRENT_LIST_DIR}/common"
		"${CMAKE_CURRENT_LIST_DIR}/histogram"
		"${CMAKE_CURRENT_LIST_DIR}/weighted-GMM"
		"${CMAKE_CURRENT_LIST_DIR}/histogram-GMM-compressor")

if(HIP_ON)
	target_include_directories(gpu_weighted_gmm INTERFACE
		"${CMAKE_CURRENT_LIST_DIR}/hipifly")
	target_compile_definitions(gpu_weighted_gmm INTERFACE HIPIFLY)
	target_link_libraries(gpu_weighted_gmm INTERFACE hip::device hip::host)
	target_compile_options(gpu_weighted_gmm INTERFACE -fgpu-rdc)
	target_link_options(gpu_weighted_gmm INTERFACE -fgpu-rdc --hip-link)
endif()
