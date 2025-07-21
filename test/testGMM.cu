#include <iostream>
#include <new>
#include <array>
#include <cmath>

#include "cudaGMM.cuh"

using Vec3 = std::array<double, 3>;

// compute spherical Gaussian weight for 'point' with given 'mean' and 'variance'
inline double gaussianWeight(const Vec3& point, const Vec3& mean, double variance) {
    double dx = point[0] - mean[0];
    double dy = point[1] - mean[1];
    double dz = point[2] - mean[2];
    double sqNorm = dx*dx + dy*dy + dz*dz;
    return std::exp(-sqNorm / (2.0 * variance));
}

int main() {
    int resolution = 50;                // number of points per axis
    int nPoints   = resolution * resolution * resolution;
    double range[3] = {1.0, 1.0, 1.0};  // [0,1]^3 cube
    double variance = 0.005;            // controls Gaussian spread

    // three non-overlapping Gaussian centers
    Vec3 means[3] = {
        {0.25, 0.25, 0.25},
        {0.75, 0.25, 0.75},
        {0.25, 0.75, 0.75}
    };

    // aligned arrays for positions and weights
    double* posX  = new(std::align_val_t(64)) double[nPoints];
    double* posY  = new(std::align_val_t(64)) double[nPoints];
    double* posZ  = new(std::align_val_t(64)) double[nPoints];
    double* weight = new(std::align_val_t(64)) double[nPoints];

    // fill grid and compute weight = max over three Gaussians
    int idx = 0;
    for (int xIdx = 0; xIdx < resolution; ++xIdx) {
        for (int yIdx = 0; yIdx < resolution; ++yIdx) {
            for (int zIdx = 0; zIdx < resolution; ++zIdx) {
                Vec3 point = {
                    xIdx * range[0] / (resolution - 1),
                    yIdx * range[1] / (resolution - 1),
                    zIdx * range[2] / (resolution - 1)
                };
                posX[idx] = point[0];
                posY[idx] = point[1];
                posZ[idx] = point[2];

                double w0 = gaussianWeight(point, means[0], variance);
                double w1 = gaussianWeight(point, means[1], variance);
                double w2 = gaussianWeight(point, means[2], variance);
                weight[idx] = std::max(w0, std::max(w1, w2));

                ++idx;
            }
        }
    }


    // GPU data
    double *d_posX, *d_posY, *d_posZ, *d_weight;
    cudaErrChk(cudaMalloc((void**)&d_posX, nPoints * sizeof(double)));
    cudaErrChk(cudaMalloc((void**)&d_posY, nPoints * sizeof(double)));
    cudaErrChk(cudaMalloc((void**)&d_posZ, nPoints * sizeof(double)));
    cudaErrChk(cudaMalloc((void**)&d_weight, nPoints * sizeof(double)));
    cudaErrChk(cudaMemcpy(d_posX, posX, nPoints * sizeof(double), cudaMemcpyHostToDevice));
    cudaErrChk(cudaMemcpy(d_posY, posY, nPoints * sizeof(double), cudaMemcpyHostToDevice));
    cudaErrChk(cudaMemcpy(d_posZ, posZ, nPoints * sizeof(double), cudaMemcpyHostToDevice));
    cudaErrChk(cudaMemcpy(d_weight, weight, nPoints * sizeof(double), cudaMemcpyHostToDevice));

    double* dataPtr[3] = {d_posX, d_posY, d_posZ};

    // GMM initialization
    auto gmm = weightedGMM::GMM<double, 3, double>();

    double weightArray[3] = {0.2, 0.5, 0.3}; // initial weights
    double meanArray[9] = {
        0.20, 0.24, 0.29,
        0.72, 0.29, 0.79,
        0.23, 0.71, 0.78
    };

    double coVarianceMatrix[9 * 3] = {
        0.005, 0.0, 0.0,
        0.0, 0.005, 0.0,
        0.0, 0.0, 0.005,

        0.005, 0.0, 0.0,
        0.0, 0.005, 0.0,
        0.0, 0.0, 0.005,

        0.005, 0.0, 0.0,
        0.0, 0.005, 0.0,
        0.0, 0.0, 0.005
    };



    weightedGMM::GMMParam_t<double> GMMParam = {
        .numComponents = 3,
        .maxIteration = 300,
        .threshold = 10e-3,
        .weightInit = weightArray,
        .meanInit = meanArray,
        .coVarianceInit = coVarianceMatrix
    };

    auto GMMData = weightedGMM::GMMDataMultiDim<double, 3, double>(nPoints, dataPtr, d_weight);

    gmm.config(&GMMParam, &GMMData);

    // gmm.preProcessDataGMM(meanArray, maxVelocityArray);

    auto convergStep = gmm.initGMM();

    auto result = gmm.getGMMResult(0, convergStep);

    std::cout << "GMM converged in " << convergStep << " steps.\n";
    std::cout << "Weights: ";
    for (int i = 0; i < result.numComponents; ++i) {
        std::cout << result.weight[i] << " ";
    }
    std::cout << "\nMeans:\n";
    for (int i = 0; i < result.numComponents; ++i) {
        std::cout << "Component " << i << ": "
                  << result.mean[i * 3] << ", "
                  << result.mean[i * 3 + 1] << ", "
                  << result.mean[i * 3 + 2] << "\n";
    }
    std::cout << "Covariances:\n";
    for (int i = 0; i < result.numComponents; ++i) {
        std::cout << "Component " << i << ": "
                  << result.coVariance[i * 9] << ", "
                  << result.coVariance[i * 9 + 1] << ", "
                  << result.coVariance[i * 9 + 2] << ", "
                  << result.coVariance[i * 9 + 3] << ", "
                  << result.coVariance[i * 9 + 4] << ", "
                  << result.coVariance[i * 9 + 5] << ", "
                  << result.coVariance[i * 9 + 6] << ", "
                  << result.coVariance[i * 9 + 7] << ", "
                  << result.coVariance[i * 9 + 8] << "\n";
    }       


    bool pass = true;

    double tolerance = 1e-2; // tolerance for weight and mean comparison

    // check if the results are as expected
    if (std::abs(result.weight[0] - 0.33) > tolerance ||
        std::abs(result.weight[1] - 0.33) > tolerance ||
        std::abs(result.weight[2] - 0.34) > tolerance) {
        std::cout << "Weight mismatch!\n";
        pass = false;
    }

    for (int i = 0; i < 3; ++i) {
        if (std::abs(result.mean[i * 3] - means[i][0]) > tolerance ||
            std::abs(result.mean[i * 3 + 1] - means[i][1]) > tolerance ||
            std::abs(result.mean[i * 3 + 2] - means[i][2]) > tolerance) {
            std::cout << "Mean mismatch for component " << i << "!\n";
            pass = false;
        }
    }



    delete[] posX;
    delete[] posY;
    delete[] posZ;
    delete[] weight;

    cudaErrChk(cudaFree(d_posX));
    cudaErrChk(cudaFree(d_posY));
    cudaErrChk(cudaFree(d_posZ));
    cudaErrChk(cudaFree(d_weight));


    return !pass;
}