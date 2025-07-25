#ifndef _HISTOGRAM_CONFIG_H_
#define _HISTOGRAM_CONFIG_H_

#include <string>
#include "cudaTypeDef.cuh"

namespace particleHistogram::config {

    // Histogram configuration

    // For 2D
    inline constexpr int PARTICLE_HISTOGRAM2D_RES_1 = 100; // must be multiply of PARTICLE_HISTOGRAM_TILE
    inline constexpr int PARTICLE_HISTOGRAM2D_RES_2 = 100;
    inline constexpr int PARTICLE_HISTOGRAM2D_TILE = 100;
    inline constexpr int PARTICLE_HISTOGRAM2D_SIZE = PARTICLE_HISTOGRAM2D_RES_1 * PARTICLE_HISTOGRAM2D_RES_2;

    // For 3D
    inline constexpr int PARTICLE_HISTOGRAM3D_RES_1 = 100; 
    inline constexpr int PARTICLE_HISTOGRAM3D_RES_2 = 100;
    inline constexpr int PARTICLE_HISTOGRAM3D_RES_3 = 100;
    inline constexpr int PARTICLE_HISTOGRAM3D_SIZE = PARTICLE_HISTOGRAM3D_RES_1 * PARTICLE_HISTOGRAM3D_RES_2 * PARTICLE_HISTOGRAM3D_RES_3;
    inline constexpr int PARTICLE_HISTOGRAM3D_TILE = 20;

    inline constexpr int PARTICLE_HISTOGRAM_MAX_SMEM = 48 * 1024; // 48KB

    inline constexpr bool HISTOGRAM_FIXED_RANGE = true; 
    inline constexpr cudaCommonType MIN_VELOCITY_HIST_E = -0.2;
    inline constexpr cudaCommonType MAX_VELOCITY_HIST_E = 0.2;
    inline constexpr cudaCommonType MIN_VELOCITY_HIST_I = -0.09;
    inline constexpr cudaCommonType MAX_VELOCITY_HIST_I = 0.09;

} // namespace particleHistogram::config


#endif