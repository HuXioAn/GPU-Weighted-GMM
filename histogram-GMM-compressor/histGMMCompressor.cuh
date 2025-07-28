#include <string>
#include <memory>
#include <random>
#include <random>
#include <vector>

#include "histogram.cuh"
#include "cudaGMM.cuh"

/**
 * TO DO 
 * Implement a class to provide a single interface for the entire histogram+GMM compression scheme
 */
namespace histogramGMMCompressor {


template<int DIM>
struct DefaultNumData {
    static_assert(DIM == 2 || DIM == 3, "Only 2D or 3D supported");
    static constexpr int value = [](){
        if constexpr (DIM == 2) {
            return particleHistogram::config::PARTICLE_HISTOGRAM2D_RES_1 * 
                    particleHistogram::config::PARTICLE_HISTOGRAM2D_RES_2;
        } else {
            return particleHistogram::config::PARTICLE_HISTOGRAM3D_RES_1 * 
                    particleHistogram::config::PARTICLE_HISTOGRAM3D_RES_2 * 
                    particleHistogram::config::PARTICLE_HISTOGRAM3D_RES_3;
        }
    }();
};
template<int DIM>
constexpr int DefaultNumData<DIM>::value;



template<typename DataType, int DIM, bool storeGMMresultArray, typename WeightType = int>
class HistGMMCompressor{

private:
    std::unique_ptr<particleHistogram::ParticleHistogram<DIM>> particleHistogram_;
    std::unique_ptr<weightedGMM::GMM<DataType, DIM, WeightType>> gmm_;
    std::vector<weightedGMM::GMMResult<DataType, DIM>> gmmResultArray_;

    const int numData_ =  DefaultNumData<DIM>::value;
    const int numComponentGMM_;
    const int maxIterationGMM_;
    const DataType thresholdGMM_;
    std::vector<DataType> weightInit_;
    std::vector<DataType> meanInit_;
    std::vector<DataType> coVarianceInit_;
    int convergStepLastGMM_;
    int simulationStepLast_;


    std::mt19937                          rng_;
    std::uniform_real_distribution<DataType>    uni01_;
    std::uniform_real_distribution<DataType>    uniTheta_;
    std::normal_distribution<DataType>          normDist_;

public:
    /**
     * @param histInitSize   initial buffer size for the histogram
     * @param numComponents  number of mixtures for the GMM
    */
    HistGMMCompressor(const int histInitSize, const int numComponentGMM, const int maxIterationGMM, const DataType thresholdGMM ) : 
        particleHistogram_(std::make_unique<particleHistogram::ParticleHistogram<DIM>>(histInitSize)),
        gmm_(std::make_unique<weightedGMM::GMM<DataType, DIM, WeightType>>() ),
        numComponentGMM_(numComponentGMM),
        maxIterationGMM_(maxIterationGMM_),
        thresholdGMM_(thresholdGMM),
        rng_(std::random_device{}()),
        uni01_(DataType(0), DataType(1)),
        normDist_(DataType(0), DataType(1))
        {
            weightInit_.resize(numComponentGMM_);
            meanInit_.resize(numComponentGMM_ * DIM);
            coVarianceInit_.resize(numComponentGMM_ * DIM * DIM);
        }

    ~HistGMMCompressor() = default;

    /// @brief Get a non-owning pointer to the underlying ParticleHistogram
    __host__ particleHistogram::ParticleHistogram<DIM>* getParticleHistogramPtr() noexcept {
        return particleHistogram_.get();
    }
    /// @brief (const) Get a non-owning pointer to the underlying ParticleHistogram
    __host__ const particleHistogram::ParticleHistogram<DIM>* getParticleHistogramPtr() const noexcept {
        return particleHistogram_.get();
    }

    /// @brief Get a non-owning pointer to the underlying GMM
    __host__ weightedGMM::GMM<DataType, DIM, WeightType>* getGMMPtr() noexcept {
        return gmm_.get();
    }
    /// @brief (const) Get a non-owning pointer to the underlying GMM
    __host__ const weightedGMM::GMM<DataType, DIM, WeightType>* getGMMPtr() const noexcept {
        return gmm_.get();
    }

    /**
     * this is a critical function, it must be tuned accoridg to the proper needings, based on the data to compress
     */
    __host__ void setGMMInitialParameters()
    {
        const DataType maxVelocity = 1;
        for(int j = 0; j < numComponentGMM_; j++){
            weightInit_[j] = 1.0/numComponentGMM_;
            const DataType n1 = normDist_(rng_);
            const DataType n2 = normDist_(rng_);
            DataType n3 = normDist_(rng_);
            if constexpr(DIM == 2){ n3 = 0; }
            const DataType u = uni01_(rng_);
            const DataType llsqrt = sqrt( n1*n1 + n2*n2 + n3*n3); 
            meanInit_[j * DIM] = maxVelocity * std::cbrt(u) * n1 / llsqrt; 
            meanInit_[j * DIM + 1] = maxVelocity * std::cbrt(u) * n2 / llsqrt;
            if constexpr (DIM == 3){ meanInit_[j * DIM + 2] = maxVelocity * std::cbrt(u) * n3 / llsqrt;}

            for (int k = 0; k < DIM * DIM; k++){
                const int i = k / DIM; // row index
                const int l = k % DIM; // column index
                coVarianceInit_[j * DIM * DIM + k] = (i == l) ? 0.02 : 0.0 ;
            }
        }
        
    }

    template<int D = DIM, typename = std::enable_if_t<D == 2>>
    __host__ void runCompression(DataType* xArrayDevicePtr, DataType* yArrayDevicePtr, DataType* qArrayDevicePtr, const int pclNum, const int species, const int simulationStep, cudaStream_t stream = 0)
    {
        runHistogram(xArrayDevicePtr, yArrayDevicePtr, qArrayDevicePtr, pclNum, species, stream);
        cudaErrChk(cudaDeviceSynchronize());        
        setGMMInitialParameters();
        weightedGMM::GMMParam_t<DataType> GMMParam = {
            .numComponents = numComponentGMM_,
            .maxIteration = maxIterationGMM_,
            .threshold = thresholdGMM_,
            .weightInit = weightInit_.data(),
            .meanInit = meanInit_.data(),
            .coVarianceInit = coVarianceInit_.data()
        };
        DataType* dataPtr[2] = {xArrayDevicePtr, yArrayDevicePtr};
        auto GMMData = weightedGMM::GMMDataMultiDim<DataType, DIM, WeightType>(numData_, particleHistogram_->getHistogramScaleMark(), particleHistogram_->getParticleHistogramCUDAArray());

        gmm_->config(&GMMParam, &GMMData);
        convergStepLastGMM_ = gmm_->initGMM();
        simulationStepLast_ = simulationStep;
        if constexpr (storeGMMresultArray) {gmmResultArray_.push_back( gmm_->getGMMResult(simulationStep, convergStepLastGMM_) );}


    }
    template<int D = DIM, typename = std::enable_if_t<D == 3>>
    __host__ void runCompression(DataType* xArrayDevicePtr, DataType* yArrayDevicePtr, DataType* zArrayDevicePtr, DataType* qArrayDevicePtr, const int pclNum, const int species, cudaStream_t stream = 0)
    {
        runHistogram(xArrayDevicePtr, yArrayDevicePtr, zArrayDevicePtr, qArrayDevicePtr, pclNum, species, stream);
        cudaErrChk(cudaDeviceSynchronize()); 
    }



    template<int D = DIM, typename = std::enable_if_t<D == 2>>
    __host__ inline void runHistogram(DataType* xArrayDevicePtr, DataType* yArrayDevicePtr, DataType* qArrayDevicePtr, const int pclNum, const int species, cudaStream_t stream = 0)
    {
        particleHistogram_->init(xArrayDevicePtr, yArrayDevicePtr, qArrayDevicePtr, pclNum, species, stream);
        cudaErrChk(cudaDeviceSynchronize()); 
    }
    template<int D = DIM, typename = std::enable_if_t<D == 3>>
    __host__ inline void runHistogram(DataType* xArrayDevicePtr, DataType* yArrayDevicePtr, DataType* zArrayDevicePtr, DataType* qArrayDevicePtr, const int pclNum, const int species, cudaStream_t stream = 0)
    {
        particleHistogram_->init(xArrayDevicePtr, yArrayDevicePtr, zArrayDevicePtr, qArrayDevicePtr, pclNum, species, stream);
        cudaErrChk(cudaDeviceSynchronize()); 
    }


    __host__ weightedGMM::GMMResult<DataType, DIM> getLastResultGMM()
    {
        return gmm_->getGMMResult(simulationStepLast_, convergStepLastGMM_);
    }

    __host__ void writeResultGMM(const std::string outputPath, const std::string metaData = "")
    {
        if (gmmResultArray_.empty()){
            std::cerr<< "Warning! Attempted writing GMM results but gmmResultArray_ is empty" <<std::endl;
        }
        else{
            gmmResultArray_[0].outputResultArray(gmmResultArray_,outputPath, metaData);
        }
    }

};

}