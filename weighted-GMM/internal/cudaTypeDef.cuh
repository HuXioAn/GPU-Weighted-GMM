#ifndef _CUDA_TYPE_DEF_H_
#define _CUDA_TYPE_DEF_H_

#ifndef HIPIFLY
#include <cuda.h>
#include "cuda_fp16.h"
#define WARP_SIZE (32)
inline constexpr uint32_t WARP_FULL_MASK = 0xFFFFFFFF;
#else
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include "hipifly.hpp"
#define WARP_SIZE (64)
inline constexpr uint64_t WARP_FULL_MASK = 0xFFFFFFFFFFFFFFFF;
#endif

#include <iostream>
#include <sstream>

namespace weightedGMM::internal {

    using cudaTypeSingle = float;
    using cudaTypeDouble = double;
    using cudaTypeHalf = __half;

    using cudaCommonType = cudaTypeDouble;


    /////////////////////////////////// CUDA API HOST call wrapper

    #define ERROR_CHECK_C_LIKE false


#define cudaErrChk(call) internal::cudaCheck((call), __FILE__, __LINE__)

    __host__ inline void cudaCheck(cudaError_t code, const char *file, int line)
    {
        if (code != cudaSuccess)
        {
    #if ERROR_CHECK_C_LIKE == true
            std::cerr << "CUDA Check: " << cudaGetErrorString(code) << " File: " << file << " Line: " << line << std::endl;
            abort();
    #else
            std::ostringstream oss;
            oss << "CUDA Check: " << cudaGetErrorString(code) << " File: " << file << " Line: " << line;
            throw std::runtime_error(oss.str());
    #endif
        }
    }
    #undef ERROR_CHECK_C_LIKE

    ////////////////////////////////// Round up to
    template <typename T>
    __host__ __device__ inline T getGridSize(T threadNum, T blockSize) {
        return ((threadNum + blockSize - 1) / blockSize);
    }
}

#endif