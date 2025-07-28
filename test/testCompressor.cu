#include <cmath>
#include <random>
#include <vector>
#include <cassert>

#include "histGMMCompressor.cuh"

using DataType = cudaCommonType;
using Vec  = std::vector<DataType>;
using Mat  = std::vector<Vec>;  // row‑major

constexpr int nSample = 100000;
constexpr int DIM = 2;

// Compute lower‑triangular L such that A = L * Lᵀ
bool cholesky(const Mat& A, Mat& L) {
    int N = A.size();
    L.assign(N, Vec(N, 0.0));
    for(int i = 0; i < N; ++i) {
        for(int j = 0; j <= i; ++j) {
            DataType sum = 0;
            for(int k = 0; k < j; ++k)
                sum += L[i][k] * L[j][k];
            if(i == j) {
                DataType diag = A[i][i] - sum;
                if(diag <= 0) return false;  // not PD
                L[i][j] = std::sqrt(diag);
            } else {
                L[i][j] = (A[i][j] - sum) / L[j][j];
            }
        }
    }
    return true;
}

// Multiply L * z
Vec matVec(const Mat& L, const Vec& z) {
    int N = L.size();
    Vec x(N, 0.0);
    for(int i = 0; i < N; ++i) {
        for(int j = 0; j <= i; ++j)
            x[i] += L[i][j] * z[j];
    }
    return x;
}

int main() {
    // Example 2‑D
    // RNG setup
    std::mt19937 rng(std::random_device{}());
    std::normal_distribution<DataType> dist(0.0, 1.0);
    const int NGaussians = 4;
    std::vector<Vec> mu    = { {0.3, 0.2}, {0.4,-0.1}, {-0.1,-0.2}, {0.3,0.6} };
    std::vector<Mat> Sigma = {{{0.06, 0.01},
                               {0.01, 0.02}},
                              {{0.05, 0.0},
                               {0.0, 0.03}},
                              {{0.007, 0.0002},
                               {0.01, 0.05}},
                               {{0.009, 0.0},
                               {0.0, 0.02}}
                            };

    const int nop = nSample * NGaussians;
    auto uCPU = new (std::align_val_t(64))DataType[nop];
    auto vCPU = new (std::align_val_t(64))DataType[nop];
    auto qCPU = new (std::align_val_t(64))DataType[nop];

    for(int i = 0; i< NGaussians; i++){
        Mat L;
        assert(cholesky(Sigma[i], L) && "Covariance not PD!");
        std::cout << "Init sampling gaussian " << i+1 << std::endl;
        for(int j = 0; j < nSample; j++){
            bool sample = true;
            while (sample){
                Vec z(DIM);
                for(int k = 0; k < DIM; ++k)
                    z[k] = dist(rng);

                Vec x = matVec(L, z);
                for(int k = 0; k < DIM; ++k){
                    x[k] += mu[i][k];
                }
                if ( x[0] > -1.0 && x[0] < 1.0 && x[1] > -1.0 && x[1] < 1.0 ){
                    sample = false;
                    uCPU[j + i*nSample] = x[0];
                    vCPU[j + i*nSample] = x[1];
                    qCPU[j + i*nSample] = 1e-6;
                }
            }
        }
    }

    // histogram CPU
    const int histogramSize2D = particleHistogram::config::PARTICLE_HISTOGRAM2D_RES_1 * particleHistogram::config::PARTICLE_HISTOGRAM2D_RES_2;
    
    std::vector<cudaCommonType> cpuHist(histogramSize2D, 0);

    cudaCommonType minVal = particleHistogram::config::MIN_VELOCITY_HIST_E;
    cudaCommonType maxVal = particleHistogram::config::MAX_VELOCITY_HIST_E;
    cudaCommonType resolution1 = (maxVal - minVal) / particleHistogram::config::PARTICLE_HISTOGRAM2D_RES_1;
    cudaCommonType resolution2 = (maxVal - minVal) / particleHistogram::config::PARTICLE_HISTOGRAM2D_RES_2;

    for (int i = 0; i < nop; i++){
        cudaCommonType uVal = uCPU[i];
        cudaCommonType vVal = vCPU[i];

        if(uVal >= minVal && uVal <= maxVal && vVal >= minVal && vVal <= maxVal){
            int bin1 = static_cast<int>((uVal - minVal) / resolution1);
            if(bin1 >= particleHistogram::config::PARTICLE_HISTOGRAM3D_RES_1) bin1 = particleHistogram::config::PARTICLE_HISTOGRAM3D_RES_1 - 1;
            int bin2 = static_cast<int>((vVal - minVal) / resolution2);
            if(bin2 >= particleHistogram::config::PARTICLE_HISTOGRAM3D_RES_2) bin2 = particleHistogram::config::PARTICLE_HISTOGRAM3D_RES_2 - 1;

            cpuHist[bin1 + bin2 * particleHistogram::config::PARTICLE_HISTOGRAM2D_RES_1] += std::abs(qCPU[i] * 1e7); // 10e6 in the kernel 
        }

    }

    // copy data to GPU

    DataType* uPtr;
    DataType* vPtr;
    DataType* qPtr;
    
    cudaErrChk(cudaMalloc((void**)&uPtr, nop * sizeof(DataType)));
    cudaErrChk(cudaMalloc((void**)&vPtr, nop * sizeof(DataType)));
    cudaErrChk(cudaMalloc((void**)&qPtr, nop * sizeof(DataType)));

    cudaErrChk(cudaMemcpy(uPtr, uCPU, nop * sizeof(DataType), cudaMemcpyHostToDevice));
    cudaErrChk(cudaMemcpy(vPtr, vCPU, nop * sizeof(DataType), cudaMemcpyHostToDevice));
    cudaErrChk(cudaMemcpy(qPtr, qCPU, nop * sizeof(DataType), cudaMemcpyHostToDevice));


    // x now ~ N(mu, Sigma)
    const int nComponentGMM = 4;
    const int maxIterGMM = 200;
    const DataType thresholdGMM = 1e-3;
    histogramGMMCompressor::HistGMMCompressor<DataType,2,true,cudaTypeSingle> compressor(histogramSize2D,nComponentGMM,maxIterGMM,thresholdGMM);
    compressor.runHistogram(uPtr, vPtr, qPtr, nSample * NGaussians , 0, 0);

    cudaErrChk(cudaDeviceSynchronize());

    auto* histogram = compressor.getParticleHistogramPtr();
    histogram->copyHistogramToHost();
    auto histogramHostPtr = histogram->getParticleHistogramHostPtr();

    // compare the results
    bool pass = true;
    cudaCommonType tolerance = 1e-1;

    for (int i = 0; i < histogramSize2D; i++){
        if (std::fabs(histogramHostPtr[i] - cpuHist[i]) > tolerance){
            std::cout << "Mismatch in UV histogram at bin " << i 
                      << ": GPU = " << histogramHostPtr[i] 
                      << ", CPU = " << cpuHist[i] << " Mismatch = "<< fabs(histogramHostPtr[i] - cpuHist[i]) << "\n";
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

    
    compressor.runCompression(uPtr, vPtr, qPtr, nSample * NGaussians , 0, 0);
    compressor.writeResultGMM("test");
    return 0;
}
