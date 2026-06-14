# GPUWeightedGMMConfig.cmake
# Defines the imported interface target 'gpu_weighted_gmm'.
#
# Usage from a consumer project:
#
#   set(GPUWeightedGMM_DIR "/path/to/GPU-Weighted-GMM")
#   find_package(GPUWeightedGMM REQUIRED)
#   target_link_libraries(my_target PRIVATE gpu_weighted_gmm)

# Guard: only define the target once even if find_package is called multiple times.
if(TARGET gpu_weighted_gmm)
	return()
endif()

get_filename_component(_GPUWGMM_ROOT "${CMAKE_CURRENT_LIST_DIR}" ABSOLUTE)

add_library(gpu_weighted_gmm INTERFACE IMPORTED GLOBAL)

target_include_directories(gpu_weighted_gmm
	INTERFACE
		"${_GPUWGMM_ROOT}/common"
		"${_GPUWGMM_ROOT}/histogram"
		"${_GPUWGMM_ROOT}/weighted-GMM"
		"${_GPUWGMM_ROOT}/histogram-GMM-compressor")

if(HIP_ON)
	target_include_directories(gpu_weighted_gmm INTERFACE "${_GPUWGMM_ROOT}/hipifly")
	target_compile_definitions(gpu_weighted_gmm INTERFACE HIPIFLY)
	target_link_libraries(gpu_weighted_gmm INTERFACE hip::device hip::host)
	target_compile_options(gpu_weighted_gmm INTERFACE -fgpu-rdc)
	target_link_options(gpu_weighted_gmm INTERFACE -fgpu-rdc --hip-link)
endif()

unset(_GPUWGMM_ROOT)
