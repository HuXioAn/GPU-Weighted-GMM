#ifndef _WEIGHTED_GMM_CONFIG_H_
#define _WEIGHTED_GMM_CONFIG_H_

#include <string>
#include "cudaTypeDef.cuh"

namespace weightedGMM::config {

inline constexpr bool NORMALIZE_DATA_FOR_GMM = true;    // normalize data before GMM such that velocities are in range -1;1 --> the original velocity domain is assumed to be symmetric wrt 0
inline constexpr bool CHECK_COVMATRIX_GMM = true;   // safety check on the cov-matrix --> ensures variances > EPS_COVMATRIX_GMM
inline constexpr internal::cudaCommonType TOL_COVMATRIX_GMM = 1e-9;  // tol used to ensure cov-matrix determinant > 0
inline constexpr internal::cudaCommonType EPS_COVMATRIX_GMM = 1e-4;   // minimum value that elements on the cov-matrix main diagonal can assume (assume data normalized in range -1,1)
inline constexpr bool PRUNE_COMPONENTS_GMM = true; // remove GMM components with weight < PRUNE_THRESHOLD_GMM --> remove one componet at a time
inline constexpr internal::cudaCommonType PRUNE_THRESHOLD_GMM = 0.005;

} // namespace weightedGMM::config


#endif