/******************************************************************************
 * Author      : Andong Hu (hu7006@outlook.com)
 * Affiliation : KTH Royal Institute of Technology
 * Time        : 2024-2025
 *
 * Notes       : This code is developed for the iPIC3D-GPU project, for the GMM data analysis pipeline.
 *              https://github.com/iPIC3D/iPIC3D-GPU
 ******************************************************************************/

#include "histogram.cuh"

#include <iostream>
#include "cudaTypeDef.cuh"
#include "cudaReduction.cuh"
#include "histogramConfig.cuh"


namespace particleHistogram
{

using namespace histogram;

__global__ void histogramKernel3D(const int nop, const histogramTypeIn *d1, const histogramTypeIn *d2, const histogramTypeIn *d3, 
                                    const histogramTypeIn *q,
                                    particleHistogramCUDA3D *histogramCUDAPtr)
{

    int pidx = threadIdx.x + blockIdx.x * blockDim.x;
    int gridSize = gridDim.x * blockDim.x;

    extern __shared__ histogramTypeOut sHistogram[];
    auto gHistogram = histogramCUDAPtr[0].getHistogramCUDA();

    constexpr int tile = config::PARTICLE_HISTOGRAM3D_TILE;
    constexpr auto tileSize = tile * tile * tile;

    histogramTypeIn data[3];
    int dim[3];
    int dimSize[3] = {tile, tile, tile};
    histogramTypeOut qAbs;

    // size of this histogram
    const auto dim0Size = histogramCUDAPtr[0].getSize(0);
    const auto dim1Size = histogramCUDAPtr[0].getSize(1);
    const auto dim2Size = histogramCUDAPtr[0].getSize(2);

    // the dim sizes are multiply of tile
    for (int dim0 = 0; dim0 < dim0Size; dim0 += tile) // unroll if const size
    for (int dim1 = 0; dim1 < dim1Size; dim1 += tile)
    for (int dim2 = 0; dim2 < dim2Size; dim2 += tile)
    {
        dim[0] = dim0;
        dim[1] = dim1;
        dim[2] = dim2;

        // Initialize shared memory to zero
        for (int i = threadIdx.x; i < tileSize; i += blockDim.x)
        {
            sHistogram[i] = 0.0;
        }
        __syncthreads();

        for (int i = pidx; i < nop; i += gridSize)
        {
            data[0] = d1[i];
            data[1] = d2[i];
            data[2] = d3[i];
            const auto sIndex = histogramCUDAPtr[0].getIndexTiled(data, dim, dimSize);
            if (sIndex < 0) continue; // out of tile range

            qAbs = abs(q[i] * 10e6);
            atomicAdd(&sHistogram[sIndex], qAbs);
        }

        __syncthreads();

        for (int i = threadIdx.x; i < tileSize; i += blockDim.x)
        {
            atomicAdd(&gHistogram[dim0 + i % tile + (dim1 + i / tile % tile) * dim0Size + (dim2 + i / tile / tile) * dim0Size * dim1Size], sHistogram[i]);
        }
    }
}

/**
 * @brief reset and calculate the center of each histogram bin
 * @details this kernel is launched once for each histogram bin for all 3 histograms
 */
__global__ void resetBin(particleHistogramCUDA3D* histogramCUDAPtr){
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int gridSize = gridDim.x * blockDim.x;

    auto histogram = histogramCUDAPtr->getHistogramCUDA();
    const auto histogramSize = histogramCUDAPtr->getLogicSize();

    for (int i = idx; i < histogramSize; i += gridSize){
        histogram[i] = 0.0;
        histogramCUDAPtr->centerOfBin(i);
    }
    
}


__host__ void particleHistogram3D::init(histogramTypeIn* xArrayDevicePtr, histogramTypeIn* yArrayDevicePtr, histogramTypeIn* zArrayDevicePtr, histogramTypeIn* qArrayDevicePtr, const int pclNum, const int species, cudaStream_t stream){

    getRange(xArrayDevicePtr, yArrayDevicePtr, zArrayDevicePtr, pclNum, species, stream);
    histogramHostPtr->setHistogram(minArray, maxArray, binThisDim);
    cudaErrChk(cudaMemcpyAsync(histogramCUDAPtr, histogramHostPtr, sizeof(particleHistogramCUDA3D), cudaMemcpyHostToDevice, stream));

    const int binNum = binThisDim[0] * binThisDim[1] * binThisDim[2];
    resetBin<<<getGridSize(binNum / 8, 256), 256, 0, stream>>>(histogramCUDAPtr);

    // shared memory size
    constexpr int tileSize = config::PARTICLE_HISTOGRAM3D_TILE * config::PARTICLE_HISTOGRAM3D_TILE * config::PARTICLE_HISTOGRAM3D_TILE;
    constexpr int sharedMemSize = sizeof(histogramTypeOut) * tileSize;
    if constexpr (sharedMemSize > config::PARTICLE_HISTOGRAM_MAX_SMEM) throw std::runtime_error("Shared memory size exceeds the limit ...");
    if(binNum % tileSize != 0) throw std::runtime_error("Adjust histogram resolution to multiply of tile ...");

    histogramKernel3D<<<getGridSize(pclNum / 128, 512), 512, sharedMemSize, stream>>>
        (pclNum, xArrayDevicePtr, yArrayDevicePtr, zArrayDevicePtr, qArrayDevicePtr,
        histogramCUDAPtr);

}


/**
 * @brief Synchronous function to get the min and max for 3 dimensions, result is stored in minArray and maxArray
 */
__host__ int particleHistogram3D::getRange(histogramTypeIn* xArrayDevicePtr, histogramTypeIn* yArrayDevicePtr, histogramTypeIn* zArrayDevicePtr, const int pclNum, const int species, cudaStream_t stream){


    if(config::HISTOGRAM_FIXED_RANGE == false){
        using namespace weightedGMM::reduction;

        constexpr int blockSize = 256;
        auto blockNum = reduceBlockNum(pclNum, blockSize);

        histogramTypeIn* pclArray[3] = {xArrayDevicePtr, yArrayDevicePtr, zArrayDevicePtr};

        for(int i=0; i<3; i++){ // UVW
            reduceMin<histogramTypeIn, blockSize><<<blockNum, blockSize, blockSize * sizeof(histogramTypeIn), stream>>>
                (pclArray[i], reductionTempArrayCUDA + i * reductionTempArraySize, pclNum);
            reduceMinWarp<histogramTypeIn><<<1, WARP_SIZE, 0, stream>>>
                (reductionTempArrayCUDA + i * reductionTempArraySize, reductionMinResultCUDA + i, blockNum);

            reduceMax<histogramTypeIn, blockSize><<<blockNum, blockSize, blockSize * sizeof(histogramTypeIn), stream>>>
                (pclArray[i], reductionTempArrayCUDA + (i+3) * reductionTempArraySize, pclNum);
            reduceMaxWarp<histogramTypeIn><<<1, WARP_SIZE, 0, stream>>>
                (reductionTempArrayCUDA + (i+3) * reductionTempArraySize, reductionMaxResultCUDA + i, blockNum);
        }
        cudaErrChk(cudaMemcpyAsync(minArray, reductionMinResultCUDA, sizeof(histogramTypeIn) * 3, cudaMemcpyDeviceToHost, stream));
        cudaErrChk(cudaMemcpyAsync(maxArray, reductionMaxResultCUDA, sizeof(histogramTypeIn) * 3, cudaMemcpyDeviceToHost, stream));
        cudaErrChk(cudaStreamSynchronize(stream));

    }else{
        histogramTypeIn min = (species == 0 || species == 2) ? config::MIN_VELOCITY_HIST_E : config::MIN_VELOCITY_HIST_I;
        minArray[0] = min;
        minArray[1] = min;
        minArray[2] = min;
        histogramTypeIn max = (species == 0 || species == 2) ? config::MAX_VELOCITY_HIST_E : config::MAX_VELOCITY_HIST_I;
        maxArray[0] = max;
        maxArray[1] = max;
        maxArray[2] = max;
    }

    return 0;

}



}















