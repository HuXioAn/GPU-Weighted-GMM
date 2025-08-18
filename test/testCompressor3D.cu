#include <cmath>
#include <random>
#include <vector>
#include <cassert>

#include "histGMMCompressor.cuh"

/**
 * Example of how to use the hist+GMM compression pipeline to compress raw particle data
 * In this test particles are sampled from N 2D gaussians
 * The 
 */


using DataType = cudaCommonType;
using Vec  = std::vector<DataType>;
using Mat  = std::vector<Vec>;

// data dimension
constexpr int DIM = 3;
// number of particles for each gaussian component
constexpr int nSample = 2000000;


/**
 * Cholesky decomposition
 * Compute lower‑triangular L such that A = L * Lᵀ
 */
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
                if(diag <= 0) return false;  // not Positive definite
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

    // set number of gaussian
    const int NGaussians = 3;
    // set gaussian mean
    std::vector<Vec> mu    = { {0.25, 0.3, 0.1}, {0.1,-0.3, -0.2}, {-0.3, 0.05, 0.0}};
    // set gaussian cov matrix, row major order
    std::vector<Mat> Sigma = {{{0.002, 0.00, 0.0},
                               {0.00, 0.003, 0.0001},
                               {0.0, 0.0001, 0.001}},
                              {{0.002, 0.0, 0.0},
                               {0.0, 0.003, 0.0},
                               {0.0, 0.0, 0.0004}},
                              {{0.001, 0.000, 0.0},
                               {0.00, 0.005, 0.0001},
                               {0.00, 0.0001, 0.002}}
                            };
    
    // set cpu data structures
    // nop = total number of particles
    const int nop = nSample * NGaussians;
    auto uCPU = new (std::align_val_t(64))DataType[nop];
    auto vCPU = new (std::align_val_t(64))DataType[nop];
    auto wCPU = new (std::align_val_t(64))DataType[nop];
    auto qCPU = new (std::align_val_t(64))DataType[nop];

    // sample data from the N gaussians
    // sample nSample from each gaussian --> each gaussian has same weight = 1 / Ngaussians
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
                //if ( x[0] > -1.0 && x[0] < 1.0 && x[1] > -1.0 && x[1] < 1.0 ){
                    sample = false;
                    uCPU[j + i*nSample] = x[0];
                    vCPU[j + i*nSample] = x[1];
                    wCPU[j + i*nSample] = x[2];
                    qCPU[j + i*nSample] = 1e-7;
                //}
            }
        }
    }

    // create the histogram on the cpu to use as reference
    const int histogramSize3D = particleHistogram::config::PARTICLE_HISTOGRAM3D_RES_1 * 
                                particleHistogram::config::PARTICLE_HISTOGRAM3D_RES_2 *
                                particleHistogram::config::PARTICLE_HISTOGRAM3D_RES_2;
    std::vector<cudaCommonType> cpuHist(histogramSize3D, 0);
    cudaCommonType minVal = particleHistogram::config::MIN_VELOCITY_HIST_E;
    cudaCommonType maxVal = particleHistogram::config::MAX_VELOCITY_HIST_E;
    cudaCommonType resolution1 = (maxVal - minVal) / particleHistogram::config::PARTICLE_HISTOGRAM2D_RES_1;
    cudaCommonType resolution2 = (maxVal - minVal) / particleHistogram::config::PARTICLE_HISTOGRAM2D_RES_2;
    cudaCommonType resolution3 = (maxVal - minVal) / particleHistogram::config::PARTICLE_HISTOGRAM3D_RES_3;

    for (int i = 0; i < nop; i++){
        cudaCommonType uVal = uCPU[i];
        cudaCommonType vVal = vCPU[i];
        cudaCommonType wVal = wCPU[i];

        if(uVal >= minVal && uVal <= maxVal && vVal >= minVal && vVal <= maxVal && wVal >= minVal && wVal <= maxVal){
            int bin1 = static_cast<int>((uVal - minVal) / resolution1);
            if(bin1 >= particleHistogram::config::PARTICLE_HISTOGRAM3D_RES_1) bin1 = particleHistogram::config::PARTICLE_HISTOGRAM3D_RES_1 - 1;
            int bin2 = static_cast<int>((vVal - minVal) / resolution2);
            if(bin2 >= particleHistogram::config::PARTICLE_HISTOGRAM3D_RES_2) bin2 = particleHistogram::config::PARTICLE_HISTOGRAM3D_RES_2 - 1;
            int bin3 = static_cast<int>((wVal - minVal) / resolution3);
            if(bin3 >= particleHistogram::config::PARTICLE_HISTOGRAM3D_RES_3) bin3 = particleHistogram::config::PARTICLE_HISTOGRAM3D_RES_3 - 1;

            cpuHist[bin1 + bin2 * particleHistogram::config::PARTICLE_HISTOGRAM3D_RES_1 + bin3 * particleHistogram::config::PARTICLE_HISTOGRAM3D_RES_1 * particleHistogram::config::PARTICLE_HISTOGRAM3D_RES_2] += 
            std::abs(qCPU[i] * 1e7); // 10e6 in the kernel 
        }
    }

    // copy data to GPU to run the hist-GMM compressor

    DataType* uPtr;
    DataType* vPtr;
    DataType* wPtr;
    DataType* qPtr;
    
    cudaErrChk(cudaMalloc((void**)&uPtr, nop * sizeof(DataType)));
    cudaErrChk(cudaMalloc((void**)&vPtr, nop * sizeof(DataType)));
    cudaErrChk(cudaMalloc((void**)&wPtr, nop * sizeof(DataType)));
    cudaErrChk(cudaMalloc((void**)&qPtr, nop * sizeof(DataType)));

    cudaErrChk(cudaMemcpy(uPtr, uCPU, nop * sizeof(DataType), cudaMemcpyHostToDevice));
    cudaErrChk(cudaMemcpy(vPtr, vCPU, nop * sizeof(DataType), cudaMemcpyHostToDevice));
    cudaErrChk(cudaMemcpy(wPtr, wCPU, nop * sizeof(DataType), cudaMemcpyHostToDevice));
    cudaErrChk(cudaMemcpy(qPtr, qCPU, nop * sizeof(DataType), cudaMemcpyHostToDevice));


    // create the compressor from the histogramGMMCompressor::HistGMMCompressor class
    const int nComponentGMM = NGaussians;
    const int maxIterGMM = 200;
    const DataType thresholdGMM = 1e-3;
    histogramGMMCompressor::HistGMMCompressor<DataType,3,true,cudaTypeSingle> compressor(histogramSize3D,nComponentGMM,maxIterGMM,thresholdGMM);
    
    // run compression pipeline hist+GMM
    compressor.runCompression(uPtr, vPtr, wPtr, qPtr, nSample * NGaussians , 0, 0);

    // write result GMM --> check manually if mean, weight and cov matrix match the input data
    std::string outputFileGMM = "testCompressor3DGMM.out";
    compressor.writeResultGMM(outputFileGMM);

    std::cout << " GMM compressor output: \n" << compressor.getLastResultGMM().outputString() << std::endl;

    // get histogram objecy from the compressor
    auto histogramHostPtr = compressor.getHistogramOutputHostPtr();
    
    // compare the results cpuHist - histogramHostPtr
    bool pass = true;
    cudaCommonType tolerance = 1e-2;
    for (int i = 0; i < histogramSize3D; i++){
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

    return 0;
}
