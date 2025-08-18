#include <string>
#include <memory>
#include <random>
#include <random>
#include <vector>

#include "histogram.cuh"
#include "cudaGMM.cuh"


namespace histogramGMMCompressor {

// struct to retrieve the number of bins in the histogram at compile time
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



/**
 * Templated class that provides a single interface for the entire histogram+GMM weighted compression scheme
 * The class contains one instance of the histogram class and one instance of the GMM class
 * DataType: datatype of the input data
 * DIM (int): dimension of the input data (only 2D and 3D supported)
 * storeGMMresultArray (bool): if true create array where GMM results are saved at each compression step
 * WeightType: datatype of the input data weight
 */
template<typename DataType, int DIM, bool storeGMMresultArray, typename WeightType = int>
class HistGMMCompressor{

private:
    std::unique_ptr<particleHistogram::ParticleHistogram<DIM>> particleHistogram_;
    std::unique_ptr<weightedGMM::GMM<DataType, DIM, WeightType>> gmm_;
    std::vector<weightedGMM::GMMResult<DataType, DIM>> gmmResultArray_;

    weightedGMM::GMMParam_t<DataType> GMMParam_;

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
     * @brief Constructor for the HistGMMCompressor class
     * @param histInitSize   initial buffer size for the histogram (should be larger than numData_)
     * @param numComponents  number of gaussians in the GMM mixture
     * @param maxIterationGMM maximum number of intenal GMM iteration 
     * @param thresholdGMM  threshold for GMM convergence
    */
    HistGMMCompressor(const int histInitSize, const int numComponentGMM, const int maxIterationGMM, const DataType thresholdGMM ) : 
        particleHistogram_(std::make_unique<particleHistogram::ParticleHistogram<DIM>>(histInitSize)),
        gmm_(std::make_unique<weightedGMM::GMM<DataType, DIM, WeightType>>() ),
        numComponentGMM_(numComponentGMM),
        maxIterationGMM_(maxIterationGMM),
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
     * @brief Initialize the GMM parameters (weights, mean and covariance) before running GMM
     * this is a critical function, it must be tuned accoridg to the data to compress
     * Hint: GMM gaussians should be initialized to cover the entire data domain
     * GMM gaussians at initialization should overlap at least as possible (different mean)
     */
    __host__ void setGMMInitialParameters()
    {
        // maximum particle velocity
        const DataType maxVelocity = 1;

        // chose between fixed or random initialization
        constexpr bool randomInitialization = false;

        if constexpr(randomInitialization){     
            // randomly sample the gaussian means from a circle of radius 1  
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
                    coVarianceInit_[j * DIM * DIM + k] = (i == l) ? 0.01*(k+1) : 0.0 ;
                }
            }
        }
        else
        {
            for(int j = 0; j < numComponentGMM_; j++){
                weightInit_[j] = 1.0/numComponentGMM_;
                const DataType factor = (j%2 == 0) ? 1 : -1;
                meanInit_[j * DIM] = maxVelocity * factor * 0.1 * (j+1) ; 
                meanInit_[j * DIM + 1] = -maxVelocity * factor * 0.07 * (j+1);
                if constexpr (DIM == 3){ meanInit_[j * DIM + 2] = maxVelocity * factor * 0.05 * (j+2);}

                for (int k = 0; k < DIM * DIM; k++){
                    const int i = k / DIM; // row index
                    const int l = k % DIM; // column index
                    coVarianceInit_[j * DIM * DIM + k] = (i == l) ? 0.01*(k+1) : 0.0 ;
                }
            }

        }

        GMMParam_.numComponents = numComponentGMM_;
        GMMParam_.maxIteration = maxIterationGMM_;
        GMMParam_.threshold = thresholdGMM_;
        GMMParam_.weightInit = weightInit_.data();
        GMMParam_.meanInit = meanInit_.data();
        GMMParam_.coVarianceInit = coVarianceInit_.data();        
    }

    /**
     * @brief Run the hist+GMM compression scheme in case of 2D data
     * @param xArrayDevicePtr  pointer to the the first input data dimension (size of pclNum) e.g. [x1,x2,x3,... xN-1]
     * @param yArrayDevicePtr  pointer to the the second input data dimension (size of pclNum) e.g. [y1,y2,y3,... yN-1]
     * @param qArrayDevicePtr pointer to the input particle weigths (size of pclNum) [q1,q2,q3,... qN-1]
     * @param species  index of the species that is being processed
     * @param simulationStep simulation step in the PIC simulation
     * @param stream cuda stream used during the data compression. Using different streams it is possible to processes different species at the same time
    */
    template<int D = DIM, typename = std::enable_if_t<D == 2>>
    __host__ void runCompression(DataType* xArrayDevicePtr, DataType* yArrayDevicePtr, DataType* qArrayDevicePtr, const int pclNum, const int species, const int simulationStep, cudaStream_t stream = 0)
    {
        if(!xArrayDevicePtr){
            std::cerr<< "xArrayDevicePtr is nullptr" <<std::endl;
            std::exit(EXIT_FAILURE);
        }
        if(!yArrayDevicePtr){
            std::cerr<< "yArrayDevicePtr is nullptr" <<std::endl;
            std::exit(EXIT_FAILURE);
        }
        if(!qArrayDevicePtr){
            std::cerr<< "qArrayDevicePtr is nullptr" <<std::endl;
            std::exit(EXIT_FAILURE);
        }
        
        runHistogram(xArrayDevicePtr, yArrayDevicePtr, qArrayDevicePtr, pclNum, species, stream);
        
        setGMMInitialParameters();

        auto GMMData = weightedGMM::GMMDataMultiDim<DataType, DIM, WeightType>(numData_, particleHistogram_->getHistogramScaleMark(), particleHistogram_->getParticleHistogramCUDAArray());
        DataType maxVelocity[2] = {1.0,1.0};
        DataType meanData[2] = {0.0,0.0};
        gmm_->config(&GMMParam_, &GMMData);
        gmm_->preProcessDataGMM(meanData,maxVelocity);
        convergStepLastGMM_ = gmm_->initGMM();
        gmm_->postProcessDataGMM(maxVelocity);
        simulationStepLast_ = simulationStep;
        if constexpr (storeGMMresultArray) {gmmResultArray_.push_back( gmm_->getGMMResult(simulationStepLast_, convergStepLastGMM_) );}

    }

    /**
     * @brief Run the hist+GMM compression scheme in case of 3D data
     * @param xArrayDevicePtr  pointer to the the first input data dimension (size of pclNum) e.g. [x1,x2,x3,... xN-1]
     * @param yArrayDevicePtr  pointer to the the second input data dimension (size of pclNum) e.g. [y1,y2,y3,... yN-1]
     * @param zArrayDevicePtr  pointer to the the second input data dimension (size of pclNum) e.g. [z1,z2,z3,... zN-1]
     * @param qArrayDevicePtr pointer to the input particle weigths (size of pclNum) [q1,q2,q3,... qN-1]
     * @param species  index of the species that is being processed
     * @param simulationStep simulation step in the PIC simulation
     * @param stream cuda stream used during the data compression. Using different streams it is possible to processes different species at the same time
    */
    template<int D = DIM, typename = std::enable_if_t<D == 3>>
    __host__ void runCompression(DataType* xArrayDevicePtr, DataType* yArrayDevicePtr, DataType* zArrayDevicePtr, DataType* qArrayDevicePtr, const int pclNum, const int species, const int simulationStep, cudaStream_t stream = 0)
    {
        if(!xArrayDevicePtr){
            std::cerr<< "xArrayDevicePtr is nullptr" <<std::endl;
            std::exit(EXIT_FAILURE);
        }
        if(!yArrayDevicePtr){
            std::cerr<< "yArrayDevicePtr is nullptr" <<std::endl;
            std::exit(EXIT_FAILURE);
        }
        if(!zArrayDevicePtr){
            std::cerr<< "zArrayDevicePtr is nullptr" <<std::endl;
            std::exit(EXIT_FAILURE);
        }
        if(!qArrayDevicePtr){
            std::cerr<< "qArrayDevicePtr is nullptr" <<std::endl;
            std::exit(EXIT_FAILURE);
        }

        runHistogram(xArrayDevicePtr, yArrayDevicePtr, zArrayDevicePtr, qArrayDevicePtr, pclNum, species, stream);
               
        setGMMInitialParameters();

        auto GMMData = weightedGMM::GMMDataMultiDim<DataType, DIM, WeightType>(numData_, particleHistogram_->getHistogramScaleMark(), particleHistogram_->getParticleHistogramCUDAArray());
        DataType maxVelocity[3] = {1.0,1.0,1.0};
        DataType meanData[3] = {0.0,0.0,0.0};
        gmm_->config(&GMMParam_, &GMMData);
        gmm_->preProcessDataGMM(meanData,maxVelocity);
        convergStepLastGMM_ = gmm_->initGMM();
        gmm_->postProcessDataGMM(maxVelocity);
        simulationStepLast_ = simulationStep;
        if constexpr (storeGMMresultArray) {gmmResultArray_.push_back( gmm_->getGMMResult(simulationStepLast_, convergStepLastGMM_) );}
    }


    /**
     * @brief Run the onlhy the histogram compression stage in case of 2D data
     * @param xArrayDevicePtr  pointer to the the first input data dimension (size of pclNum) e.g. [x1,x2,x3,... xN-1]
     * @param yArrayDevicePtr  pointer to the the second input data dimension (size of pclNum) e.g. [y1,y2,y3,... yN-1]
     * @param qArrayDevicePtr pointer to the input particle weigths (size of pclNum) [q1,q2,q3,... qN-1]
     * @param species  index of the species that is being processed
     * @param stream cuda stream used during the data compression. Using different streams it is possible to processes different species at the same time
    */
    template<int D = DIM, typename = std::enable_if_t<D == 2>>
    __host__ inline void runHistogram(DataType* xArrayDevicePtr, DataType* yArrayDevicePtr, DataType* qArrayDevicePtr, const int pclNum, const int species, cudaStream_t stream = 0)
    {
        if(!xArrayDevicePtr){
            std::cerr<< "xArrayDevicePtr is nullptr" <<std::endl;
            std::exit(EXIT_FAILURE);
        }
        if(!yArrayDevicePtr){
            std::cerr<< "yArrayDevicePtr is nullptr" <<std::endl;
            std::exit(EXIT_FAILURE);
        }
        if(!qArrayDevicePtr){
            std::cerr<< "qArrayDevicePtr is nullptr" <<std::endl;
            std::exit(EXIT_FAILURE);
        }
        particleHistogram_->init(xArrayDevicePtr, yArrayDevicePtr, qArrayDevicePtr, pclNum, species, stream);
        cudaErrChk( cudaStreamSynchronize(stream) );
    }


    /**
     * @brief Run the onlhy the histogram compression stage in case of 3D data
     * @param xArrayDevicePtr  pointer to the the first input data dimension (size of pclNum) e.g. [x1,x2,x3,... xN-1]
     * @param yArrayDevicePtr  pointer to the the second input data dimension (size of pclNum) e.g. [y1,y2,y3,... yN-1]
     * @param zArrayDevicePtr  pointer to the the second input data dimension (size of pclNum) e.g. [z1,z2,z3,... zN-1]
     * @param qArrayDevicePtr pointer to the input particle weigths (size of pclNum) [q1,q2,q3,... qN-1]
     * @param species  index of the species that is being processed
     * @param stream cuda stream used during the data compression. Using different streams it is possible to processes different species at the same time
    */
    template<int D = DIM, typename = std::enable_if_t<D == 3>>
    __host__ inline void runHistogram(DataType* xArrayDevicePtr, DataType* yArrayDevicePtr, DataType* zArrayDevicePtr, DataType* qArrayDevicePtr, const int pclNum, const int species, cudaStream_t stream = 0)
    {
        if(!xArrayDevicePtr){
            std::cerr<< "xArrayDevicePtr is nullptr" <<std::endl;
            std::exit(EXIT_FAILURE);
        }
        if(!yArrayDevicePtr){
            std::cerr<< "yArrayDevicePtr is nullptr" <<std::endl;
            std::exit(EXIT_FAILURE);
        }
        if(!zArrayDevicePtr){
            std::cerr<< "zArrayDevicePtr is nullptr" <<std::endl;
            std::exit(EXIT_FAILURE);
        }
        if(!qArrayDevicePtr){
            std::cerr<< "qArrayDevicePtr is nullptr" <<std::endl;
            std::exit(EXIT_FAILURE);
        }
        particleHistogram_->init(xArrayDevicePtr, yArrayDevicePtr, zArrayDevicePtr, qArrayDevicePtr, pclNum, species, stream);
        cudaErrChk( cudaStreamSynchronize(stream) );
    }


    /**
     * @brief copy histogram output data to host and get a pointer to the data
     */
    __host__ particleHistogram::histogramTypeOut* getHistogramOutputHostPtr()
    {   
        particleHistogram_->copyHistogramToHost();
        return particleHistogram_->getParticleHistogramHostPtr();
    }


    /**
     * @brief retrieve GMM output of the last compression step
     */
    __host__ weightedGMM::GMMResult<DataType, DIM> getLastResultGMM()
    {
        return gmm_->getGMMResult(simulationStepLast_, convergStepLastGMM_);
    }

    /**
     * @brief write GMM results stored in the gmmResultArray_
     */
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