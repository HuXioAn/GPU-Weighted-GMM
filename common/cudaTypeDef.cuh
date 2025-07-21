/******************************************************************************
 * Author      : Andong Hu (hu7006@outlook.com)
 * Affiliation : KTH Royal Institute of Technology
 * Time        : 2024-2025
 *
 * Notes       : This code is developed for the iPIC3D-GPU project.
 *              https://github.com/iPIC3D/iPIC3D-GPU
 ******************************************************************************/

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


    using cudaTypeSingle = float;
    using cudaTypeDouble = double;
    using cudaTypeHalf = __half;

    using cudaCommonType = cudaTypeDouble;


    /////////////////////////////////// CUDA API HOST call wrapper

    #define ERROR_CHECK_C_LIKE false


#define cudaErrChk(call) cudaCheck((call), __FILE__, __LINE__)

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


    ////////////////////////////////// Pinned memory allocation
    __host__ inline void* allocateHostPinnedMem(size_t typeSize, size_t num){
        void* ptr = nullptr;
        cudaErrChk(cudaHostAlloc(&ptr, typeSize*num, cudaHostAllocDefault));
        return ptr;
    }
    
    template <typename T, typename... Args>
    T* newHostPinnedObject(Args... args){
        T* ptr = (T*)allocateHostPinnedMem(sizeof(T), 1);
        return new(ptr) T(std::forward<Args>(args)...);
    }

    template <typename T>
    void deleteHostPinnedObject(T* ptr){
        ptr->~T();
        cudaErrChk(cudaFreeHost(ptr));
    }



    ////////////////////////////////// Round up to
    template <typename T>
    __host__ __device__ inline T getGridSize(T threadNum, T blockSize) {
        return ((threadNum + blockSize - 1) / blockSize);
    }

#endif