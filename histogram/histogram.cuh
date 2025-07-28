/******************************************************************************
 * Author      : Andong Hu (hu7006@outlook.com)
 * Affiliation : KTH Royal Institute of Technology
 * Time        : 2024-2025
 *
 * Notes       : This code is developed for the iPIC3D-GPU project, for the GMM data analysis pipeline.
 *              https://github.com/iPIC3D/iPIC3D-GPU
 ******************************************************************************/

#ifndef _PARTICLE_HISTOGRAM_
#define _PARTICLE_HISTOGRAM_

#pragma once
#include "cudaTypeDef.cuh"
#include "cudaReduction.cuh"

#include <array>
#include <iostream>
#include <fstream>
#include <type_traits>

#include "histogramConfig.cuh"
#include "histogramKernel.cuh"


namespace histogram{

template <typename U, int dim, typename T = int>
class HistogramCUDA {

private:
    T* hostPtr;
    T* cudaPtr;

    U* scaleMark[dim];

    int bufferSize; // the physical size of current buffer, in elements
public:
    int size[dim];  // the logic size of each dimension, in elements
private:
    int logicSize;  // the logic size of the whole histogram

    U min[dim], max[dim], resolution[dim];

public:

    /**
     * @param bufferSize the physical size of the buffer, in elements
     */
    __host__ HistogramCUDA(int bufferSize): bufferSize(bufferSize){
        allocate();
    }

    /**
     * @param min the minimum value of each dimension
     * @param max the maximum value of each dimension
     * @param resolution the resolution of each dimension
     */
    __host__ void setHistogram(U* min, U* max, int* binThisDim){
        
        for(int i=0; i<dim; i++){
            if(min[i] >= max[i] || binThisDim[i] <= 0){
                std::cerr << "[!]Invalid histogram range or binThisDim" << std::endl;
                std::cerr << "[!]min: " << min[i] << " max: " << max[i] << " binThisDim: " << binThisDim[i] << std::endl;
                return;
            }
        }

        logicSize = 1;
        for(int i=0; i<dim; i++){
            this->min[i] = min[i];
            this->max[i] = max[i];

            size[i] = binThisDim[i];
            this->resolution[i] = (max[i] - min[i]) / size[i];
            logicSize *= size[i];
        }

        if(bufferSize < logicSize){
            cudaErrChk(cudaFreeHost(hostPtr));
            cudaErrChk(cudaFree(cudaPtr));
            bufferSize = logicSize;
            allocate();
        }
        
    }


    __host__ void copyHistogramAsync(cudaStream_t stream = 0){
        cudaErrChk(cudaMemcpyAsync(hostPtr, cudaPtr, logicSize * sizeof(T), cudaMemcpyDeviceToHost, stream));
    }

    __host__ T* getHistogram(){
        return hostPtr;
    }

    __host__ __device__ T* getHistogramCUDA(){
        return cudaPtr;
    }

    __host__ U** getScaleMarkCUDAPtrs(){
        return scaleMark;
    }

    __host__ __device__ int getLogicSize(){
        return logicSize;
    }

    __host__ void getSize(int* size){
        for(int i=0; i<dim; i++){
            size[i] = this->size[i];
        }
    }

    __host__ __device__ int getSize(int index){
        return size[index];
    }

    __host__ U getMin(int index){
        return min[index];
    }

    __host__ U getMax(int index){
        return max[index];
    }

    __host__ U getResolution(int index){
        return resolution[index];
    }


    __device__ int getIndex(const U* data){
        int index = 0;
        
        for(int i=dim-1; i>=0; i--){
            // check the range
            if(data[i] < min[i] || data[i] > max[i]){return -1;}

            auto tmp = (int)((data[i] - min[i]) / resolution[i]);
            if(tmp == size[i])tmp--; // the max value
            index +=  tmp;
            if(i != 0)index *= size[i-1];
        }

        if(index >= logicSize)return -1;
        return index;
    }


    /**
     * @brief get the index of the bin, in the buffer
     * @param data the data to be histogramed, dim elements
     * @param tile the start index of the tile, dim elements
     * @param tileSize the size of the tile, dim elements
     * 
     * @return the index of the bin, in Tile
     */
    __device__ int getIndexTiled(const U* data, const int* tile, const int* tileSize){
        int index = 0;
        
        for(int i=dim-1; i>=0; i--){
            // check the range
            if(data[i] < min[i] || data[i] > max[i]){return -1;}

            auto tmp = (int)((data[i] - min[i]) / resolution[i]); // the index in the whole histogram dimension
            if(tmp == size[i])tmp--; // the max value

            if(tmp < tile[i] || tmp >= tile[i] + tileSize[i])return -1; // out of the tile range

            index += tmp - tile[i];
            if(i != 0)index *= tileSize[i-1];
        }

        auto tileBufferSize = 1;
        for (int i = 0; i < dim; i++) {
            tileBufferSize *= tileSize[i];
        }

        if(index >= tileBufferSize)return -1;
        return index;
    }

    /**
    * @brief get the center of the bin, for the scale mark, for GMM
    * @param index the index of the bin, in the buffer
    */
    __device__ void centerOfBin(int index){
        int tmp = index;
        for(int i=0; i<dim; i++){
            scaleMark[i][index] = min[i] + (tmp % size[i] + 0.5) * resolution[i];
            tmp /= size[i];
        }
    }


private:

    __host__ void allocate(){
        cudaErrChk(cudaMallocHost((void**)&hostPtr, bufferSize * sizeof(T)));
        cudaErrChk(cudaMalloc((void**)&cudaPtr, bufferSize * sizeof(T)));

        for(int i=0; i<dim; i++){
            cudaErrChk(cudaMalloc((void**)&scaleMark[i], bufferSize * sizeof(U)));
        }
    }


public:

    __host__ ~HistogramCUDA(){
        cudaErrChk(cudaFreeHost(hostPtr));
        cudaErrChk(cudaFree(cudaPtr));

        for(int i=0; i<dim; i++){
            cudaErrChk(cudaFree(scaleMark[i]));
        }
    }

};

} // histogram


namespace particleHistogram{

using histogramTypeIn = cudaCommonType;
using histogramTypeOut = cudaTypeSingle;


template<int DIM>
struct DefaultBins {
    static_assert(DIM == 2 || DIM == 3, "Only 2D or 3D supported");
    static constexpr std::array<int, DIM> value = [](){
        if constexpr (DIM == 2) {
            return std::array<int,2>{
                config::PARTICLE_HISTOGRAM2D_RES_1,
                config::PARTICLE_HISTOGRAM2D_RES_2
            };
        } else {
            // DIM==3
            return std::array<int,3>{
                config::PARTICLE_HISTOGRAM3D_RES_1,
                config::PARTICLE_HISTOGRAM3D_RES_2,
                config::PARTICLE_HISTOGRAM3D_RES_3
            };
        }
    }();
};
template<int DIM>
constexpr std::array<int,DIM> DefaultBins<DIM>::value;


/**
 * @brief Histogram for one species
 */
template<int DIM>
class ParticleHistogram
{
    using particleHistogramCUDA = histogram::HistogramCUDA<histogramTypeIn, DIM, histogramTypeOut>;

private:
    // UVW
    particleHistogramCUDA* histogramHostPtr;
    particleHistogramCUDA* histogramCUDAPtr; 

    std::array<int, DIM> binThisDim = DefaultBins<DIM>::value;

    int reductionTempArraySize = 0;
    histogramTypeIn* reductionTempArrayCUDA;
    histogramTypeIn* reductionMinResultCUDA;
    histogramTypeIn* reductionMaxResultCUDA;

    histogramTypeIn minArray[DIM];
    histogramTypeIn maxArray[DIM];


    bool bigEndian;

    int reduceBlockNum(int dataSize, int blockSize){
        constexpr int elementsPerThread = 128;
        if(dataSize < elementsPerThread)dataSize = elementsPerThread;
        auto blockNum = getGridSize(dataSize / elementsPerThread, blockSize); // 4096 elements per thread
        blockNum = blockNum > 1024 ? 1024 : blockNum;

        if(reductionTempArraySize < blockNum){
            cudaErrChk(cudaFree(reductionTempArrayCUDA));
            cudaErrChk(cudaMalloc((void**)&reductionTempArrayCUDA, sizeof(histogramTypeIn)*blockNum * 2 * DIM ));
            reductionTempArraySize = blockNum;
        }

        return blockNum;
    }


    /**
     * @brief get the Max and Min value of the given value set
     */
    template<int D = DIM, typename = std::enable_if_t<D == 2>>
    __host__ int getRange(histogramTypeIn* xArrayDevicePtr, histogramTypeIn* yArrayDevicePtr, const int pclNum, const int species, cudaStream_t stream)
    {
        if(config::HISTOGRAM_FIXED_RANGE == false){
            using namespace weightedGMM::reduction;

            constexpr int blockSize = 256;
            auto blockNum = reduceBlockNum(pclNum, blockSize);

            histogramTypeIn* pclArray[2] = {xArrayDevicePtr, yArrayDevicePtr};

            for(int i=0; i<2; i++){ // UV
                reduceMin<histogramTypeIn, blockSize><<<blockNum, blockSize, blockSize * sizeof(histogramTypeIn), stream>>>
                    (pclArray[i], reductionTempArrayCUDA + i * reductionTempArraySize, pclNum);
                reduceMinWarp<histogramTypeIn><<<1, WARP_SIZE, 0, stream>>>
                    (reductionTempArrayCUDA + i * reductionTempArraySize, reductionMinResultCUDA + i, blockNum);

                reduceMax<histogramTypeIn, blockSize><<<blockNum, blockSize, blockSize * sizeof(histogramTypeIn), stream>>>
                    (pclArray[i], reductionTempArrayCUDA + (i+2) * reductionTempArraySize, pclNum);
                reduceMaxWarp<histogramTypeIn><<<1, WARP_SIZE, 0, stream>>>
                    (reductionTempArrayCUDA + (i+2) * reductionTempArraySize, reductionMaxResultCUDA + i, blockNum);
            }
            cudaErrChk(cudaMemcpyAsync(minArray, reductionMinResultCUDA, sizeof(histogramTypeIn) * 2, cudaMemcpyDeviceToHost, stream));
            cudaErrChk(cudaMemcpyAsync(maxArray, reductionMaxResultCUDA, sizeof(histogramTypeIn) * 2, cudaMemcpyDeviceToHost, stream));
            cudaErrChk(cudaStreamSynchronize(stream));

        }else{
            histogramTypeIn min = (species == 0 || species == 2) ? config::MIN_VELOCITY_HIST_E : config::MIN_VELOCITY_HIST_I;
            minArray[0] = min;
            minArray[1] = min;
            histogramTypeIn max = (species == 0 || species == 2) ? config::MAX_VELOCITY_HIST_E : config::MAX_VELOCITY_HIST_I;
            maxArray[0] = max;
            maxArray[1] = max;
        }

        return 0;
    }

    template<int D = DIM, typename = std::enable_if_t<D == 3>>
    __host__ int getRange(histogramTypeIn* xArrayDevicePtr, histogramTypeIn* yArrayDevicePtr, histogramTypeIn* zArrayDevicePtr, const int pclNum, const int species, cudaStream_t stream)
    {
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

public:

    /**
     * @param initSize the initial size of the histogram buffer, in elements
     * @param path the path to store the output file, directory
     */
    ParticleHistogram<DIM>(const int initSize) {

        auto bufferSize = binThisDim[0] * binThisDim[1];
        if constexpr(DIM == 3){bufferSize *= binThisDim[2];}
        int allocSize = initSize;
        if(initSize < bufferSize){
            if constexpr (DIM == 2){
                std::cerr << "[!]Histogram initial size is too small: " << initSize << " vs " << binThisDim[0] << "x" << binThisDim[1] << std::endl;
            } 
            else{
                std::cerr << "[!]Histogram initial size is too small: " << initSize << " vs " << binThisDim[0] << "x" << binThisDim[1] << "x" << binThisDim[0] << std::endl;
            }
            allocSize = bufferSize;
        }

        histogramHostPtr = newHostPinnedObject<particleHistogramCUDA>(allocSize);
        cudaErrChk(cudaMalloc((void**)&histogramCUDAPtr, sizeof(particleHistogramCUDA)));


        if constexpr (particleHistogram::config::HISTOGRAM_FIXED_RANGE == false){
            reductionTempArraySize = 1024;
            cudaErrChk(cudaMalloc((void**)&reductionTempArrayCUDA, sizeof(histogramTypeIn)*reductionTempArraySize * 2 * DIM));
            cudaErrChk(cudaMalloc((void**)&reductionMinResultCUDA, sizeof(histogramTypeIn) * 2  * DIM));
            reductionMaxResultCUDA = reductionMinResultCUDA + DIM;
        } 
        
        { // check the endian
            int test = 1;
            char* ptr = reinterpret_cast<char*>(&test);
            if (*ptr == 1) {
                bigEndian = false;
            } else {
                bigEndian = true;
            }
        }
    }

    /**
     * @brief Initiate the kernels for histograming, launch the kernels
     * @details It can be invoked after Moment in the main loop, for the output and solver are on CPU
     */
    template<int D = DIM, typename = std::enable_if_t<D == 2>>
    __host__ void init(histogramTypeIn* xArrayDevicePtr, histogramTypeIn* yArrayDevicePtr, histogramTypeIn* qArrayDevicePtr, const int pclNum, const int species, cudaStream_t stream = 0)
    {
        getRange(xArrayDevicePtr, yArrayDevicePtr, pclNum, species, stream);
        histogramHostPtr->setHistogram(minArray, maxArray, binThisDim.data());
        cudaErrChk(cudaMemcpyAsync(histogramCUDAPtr, histogramHostPtr, sizeof(particleHistogramCUDA), cudaMemcpyHostToDevice, stream));

        const int binNum = binThisDim[0] * binThisDim[1];
        resetBinKernel<particleHistogramCUDA><<<getGridSize(binNum / 8, 256), 256, 0, stream>>>(histogramCUDAPtr);

        // shared memory size
        constexpr int tileSize = config::PARTICLE_HISTOGRAM2D_TILE * config::PARTICLE_HISTOGRAM2D_TILE;
        constexpr int sharedMemSize = sizeof(histogramTypeOut) * tileSize;
        if constexpr (sharedMemSize > config::PARTICLE_HISTOGRAM_MAX_SMEM) throw std::runtime_error("Shared memory size exceeds the limit ...");
        if(binNum % tileSize != 0) throw std::runtime_error("Adjust histogram resolution to multiply of tile ...");

        histogramKernel2D<histogramTypeIn,histogramTypeOut,particleHistogramCUDA><<<getGridSize(pclNum / 128, 512), 512, sharedMemSize, stream>>>
            (pclNum, xArrayDevicePtr, yArrayDevicePtr, qArrayDevicePtr, histogramCUDAPtr);

    }
    
    template<int D = DIM, typename = std::enable_if_t<D == 3>>
    __host__ void init(histogramTypeIn* xArrayDevicePtr, histogramTypeIn* yArrayDevicePtr, histogramTypeIn* zArrayDevicePtr, histogramTypeIn* qArrayDevicePtr, const int pclNum, const int species, cudaStream_t stream = 0)
    {
        
        getRange(xArrayDevicePtr, yArrayDevicePtr, zArrayDevicePtr, pclNum, species, stream);
        histogramHostPtr->setHistogram(minArray, maxArray, binThisDim.data());
        cudaErrChk(cudaMemcpyAsync(histogramCUDAPtr, histogramHostPtr, sizeof(particleHistogramCUDA), cudaMemcpyHostToDevice, stream));

        const int binNum = binThisDim[0] * binThisDim[1] * binThisDim[2];
        resetBinKernel<particleHistogramCUDA><<<getGridSize(binNum / 8, 256), 256, 0, stream>>>(histogramCUDAPtr);

        // shared memory size
        constexpr int tileSize = config::PARTICLE_HISTOGRAM3D_TILE * config::PARTICLE_HISTOGRAM3D_TILE * config::PARTICLE_HISTOGRAM3D_TILE;
        constexpr int sharedMemSize = sizeof(histogramTypeOut) * tileSize;
        if constexpr (sharedMemSize > config::PARTICLE_HISTOGRAM_MAX_SMEM) throw std::runtime_error("Shared memory size exceeds the limit ...");
        if(binNum % tileSize != 0) throw std::runtime_error("Adjust histogram resolution to multiply of tile ...");

        histogramKernel3D<histogramTypeIn,histogramTypeOut,particleHistogramCUDA><<<getGridSize(pclNum / 128, 512), 512, sharedMemSize, stream>>>
            (pclNum, xArrayDevicePtr, yArrayDevicePtr, zArrayDevicePtr, qArrayDevicePtr, histogramCUDAPtr);
    }

    /**
     * @brief Wait for the histogram data to be ready, copy the data to host
     * @details It should be invoked after a previous Init, after this, can use getparticleHistogramCUDAArray to get the data
     *         writeToFile has the same effect
     */
    void copyHistogramToHost(cudaStream_t stream = 0){        
        histogramHostPtr->copyHistogramAsync(stream);
        cudaErrChk(cudaStreamSynchronize(stream));
    }


    void writeToFile(std::string filePath, int cycleNum, cudaStream_t stream = 0){
        copyHistogramToHost(stream);
        
        std::string vtkType;
        if constexpr (std::is_same_v<histogramTypeOut, float>){
            vtkType = "float";
        } else if constexpr (std::is_same_v<histogramTypeOut, double>){
            vtkType = "double";
        } else if constexpr (std::is_same_v<histogramTypeOut, int>){
            vtkType = "int";
        } else {
            throw std::runtime_error("Unsupported histogramTypeOut");
        }

        std::ostringstream ossFileName;
        if constexpr (DIM == 2){ ossFileName << filePath << "2D_" << cycleNum << ".vtk"; }
        else { ossFileName << filePath << "3D_" << cycleNum << ".vtk"; }
        

        std::ofstream vtkFile(ossFileName.str(), std::ios::binary);
        
        vtkFile << "# vtk DataFile Version 3.0\n";
        vtkFile << "Velocity Histogram\n";
        vtkFile << "BINARY\n";  
        vtkFile << "DATASET STRUCTURED_POINTS\n";
        if constexpr (DIM == 2){
            vtkFile << "DIMENSIONS " << histogramHostPtr->size[0] << " " << histogramHostPtr->size[1] << "\n";
            vtkFile << "ORIGIN " << histogramHostPtr->getMin(0) << " " << histogramHostPtr->getMin(1) << "\n";
            vtkFile << "SPACING " << histogramHostPtr->getResolution(0) << " " << histogramHostPtr->getResolution(1) << "\n";
        }
        else{
            vtkFile << "DIMENSIONS " << histogramHostPtr->size[0] << " " << histogramHostPtr->size[1] << " " << histogramHostPtr->size[2] << "\n";
            vtkFile << "ORIGIN " << histogramHostPtr->getMin(0) << " " << histogramHostPtr->getMin(1) << " " << histogramHostPtr->getMin(2) << "\n";
            vtkFile << "SPACING " << histogramHostPtr->getResolution(0) << " " << histogramHostPtr->getResolution(1) << " " << histogramHostPtr->getResolution(2) << "\n";
        }
        vtkFile << "POINT_DATA " << histogramHostPtr->getLogicSize() << "\n";  
        vtkFile << "SCALARS scalars " << vtkType << " 1\n";  
        vtkFile << "LOOKUP_TABLE default\n";  

        auto histogramBuffer = histogramHostPtr->getHistogram();
        for (int j = 0; j < histogramHostPtr->getLogicSize(); j++) {
            histogramTypeOut value = histogramBuffer[j];

            if constexpr (sizeof(histogramTypeOut) == 4){
                if(!bigEndian)*(uint32_t*)(&value) = __builtin_bswap32(*(uint32_t*)(&value));
            } else if constexpr (sizeof(histogramTypeOut) == 8){
                if(!bigEndian)*(uint64_t*)(&value) = __builtin_bswap64(*(uint64_t*)(&value));
            }

            vtkFile.write(reinterpret_cast<char*>(&value), sizeof(histogramTypeOut));
        }

        vtkFile.close();

    }

    histogramTypeOut* getParticleHistogramHostPtr(){
        return histogramHostPtr->getHistogram();
    }

    histogramTypeOut* getParticleHistogramCUDAArray(){
        return histogramHostPtr->getHistogramCUDA();
    }

    histogramTypeIn** getHistogramScaleMark(){
        return histogramHostPtr->getScaleMarkCUDAPtrs();
    }


    ~ParticleHistogram<DIM>(){
        if constexpr (particleHistogram::config::HISTOGRAM_FIXED_RANGE == false){
            cudaErrChk(cudaFree(reductionTempArrayCUDA));
            cudaErrChk(cudaFree(reductionMinResultCUDA));
        }

        cudaErrChk(cudaFree(histogramCUDAPtr));
        deleteHostPinnedObject(histogramHostPtr);
    }
};

    
} //particleHistogram






#endif