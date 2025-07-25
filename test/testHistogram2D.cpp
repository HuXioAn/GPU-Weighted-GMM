
#include <string>
#include <memory>
#include <random>

#include "histogram.cuh"

using namespace particleHistogram;
using namespace particleHistogram::config;

constexpr int nop = 5000000;


int main(){
    int histogramSize2D = PARTICLE_HISTOGRAM2D_SIZE;

    // histogram 2D test 
    particleHistogram2D histogram(histogramSize2D);

    // fill the array with random data
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<histogramTypeIn> dis(-1.0, 1.0);
    auto center = MIN_VELOCITY_HIST_E + MAX_VELOCITY_HIST_E;
    auto left = center - MIN_VELOCITY_HIST_E;
    auto right = MAX_VELOCITY_HIST_E - center;
    std::normal_distribution<histogramTypeIn> normalDist1(center, right * 0.2);
    std::normal_distribution<histogramTypeIn> normalDist2(center + right * 0.4, right * 0.1);

    auto uCPU = new (std::align_val_t(64))histogramTypeIn[nop];
    auto vCPU = new (std::align_val_t(64))histogramTypeIn[nop];
    auto qCPU = new (std::align_val_t(64))histogramTypeIn[nop];

    for(int i = 0; i < nop; i++){
        uCPU[i] = dis(gen) > 0.0 ? normalDist1(gen) : normalDist2(gen);
        vCPU[i] = dis(gen) > 0.0 ? normalDist1(gen) : normalDist2(gen);
        qCPU[i] = dis(gen) / 1e10;
    }

    // GPU

    histogramTypeIn* uPtr;
    histogramTypeIn* vPtr;
    histogramTypeIn* qPtr;

    cudaErrChk(cudaMalloc((void**)&uPtr, nop * sizeof(histogramTypeIn)));
    cudaErrChk(cudaMalloc((void**)&vPtr, nop * sizeof(histogramTypeIn)));
    cudaErrChk(cudaMalloc((void**)&qPtr, nop * sizeof(histogramTypeIn)));

    cudaErrChk(cudaMemcpy(uPtr, uCPU, nop * sizeof(histogramTypeIn), cudaMemcpyHostToDevice));
    cudaErrChk(cudaMemcpy(vPtr, vCPU, nop * sizeof(histogramTypeIn), cudaMemcpyHostToDevice));
    cudaErrChk(cudaMemcpy(qPtr, qCPU, nop * sizeof(histogramTypeIn), cudaMemcpyHostToDevice));

    // CPU 2D histogram

    std::vector<cudaCommonType> cpuHist(histogramSize2D, 0);


    cudaCommonType minVal = MIN_VELOCITY_HIST_E;
    cudaCommonType maxVal = MAX_VELOCITY_HIST_E;
    cudaCommonType resolution1 = (maxVal - minVal) / PARTICLE_HISTOGRAM2D_RES_1;
    cudaCommonType resolution2 = (maxVal - minVal) / PARTICLE_HISTOGRAM2D_RES_2;

    for (int i = 0; i < nop; i++){
        cudaCommonType uVal = uCPU[i];
        cudaCommonType vVal = vCPU[i];

        if(uVal >= minVal && uVal <= maxVal && vVal >= minVal && vVal <= maxVal){
            int bin1 = static_cast<int>((uVal - minVal) / resolution1);
            if(bin1 >= PARTICLE_HISTOGRAM3D_RES_1) bin1 = PARTICLE_HISTOGRAM3D_RES_1 - 1;
            int bin2 = static_cast<int>((vVal - minVal) / resolution2);
            if(bin2 >= PARTICLE_HISTOGRAM3D_RES_2) bin2 = PARTICLE_HISTOGRAM3D_RES_2 - 1;

            cpuHist[bin1 + bin2 * PARTICLE_HISTOGRAM2D_RES_1] += std::abs(qCPU[i] * 1e7); // 10e6 in the kernel 
        }

    }

    // GPU histogram
    histogram.init(uPtr, vPtr, qPtr, nop, 0, 0);
    cudaErrChk(cudaDeviceSynchronize());
    histogram.copyHistogramToHost();

    auto histogramHostPtr = histogram.getParticleHistogramHostPtr();

    // compare the results
    bool pass = true;
    cudaCommonType tolerance = 1e-6;

    for (int i = 0; i < histogramSize2D; i++){
        if (std::fabs(histogramHostPtr[i] - cpuHist[i]) > tolerance){
            std::cout << "Mismatch in UV histogram at bin " << i 
                      << ": GPU = " << histogramHostPtr[i] 
                      << ", CPU = " << cpuHist[i] << "Mismatch = "<< fabs(histogramHostPtr[i] - cpuHist[i]) << "\n";
            pass = false;
            break;
        }
    }

    if(pass){
        std::cout << "Test passed: CPU and GPU histograms match.\n";
    } else {
        std::cout << "Test failed: CPU and GPU histograms do not match.\n";
    }

    delete[] uCPU;
    delete[] vCPU;
    delete[] qCPU;

    cudaErrChk(cudaFree(uPtr));
    cudaErrChk(cudaFree(vPtr));
    cudaErrChk(cudaFree(qPtr));

    return !pass;

}









