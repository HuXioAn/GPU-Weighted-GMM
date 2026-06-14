# Histogram-GMM Data Compression Scheme

A header-only C++ library implementing histogramming and Gaussian Mixture Model (GMM) data compression for particle data on accelerators, with support for both CUDA and HIP/ROCm backends.

## Features

- **Histogram Stage:** Efficient GPU-based histogramming of particle data (2D and 3D).
- **GMM Stage:** Weighted Gaussian Mixture Model fitting for data compression.
- **Full Pipeline:** Combined histogram + GMM compressor for maximum compression.
- **Modular:** Use either histogram-only, GMM-only, or full pipeline.
- **Backend Support:** Compatible with CUDA and HIP/ROCm via the `hipifly` compatibility layer.
- **Header-only:** No compilation of library sources required; integrate via CMake `find_package`.

> **Note**: GMM initialization critically affects convergence and should be tailored to your specific data.

## Folder Structure

- **common:** Shared type definitions and reduction kernels (`cudaTypeDef.cuh`, `cudaReduction.cuh`).
- **histogram:** GPU histogram class and compile-time configuration (`histogramConfig.cuh`).
- **weighted-GMM:** Weighted GMM class and compile-time configuration (`GMMConfig.cuh`).
- **histogram-GMM-compressor:** Full histogram + GMM compression pipeline (`histGMMCompressor.cuh`).
- **test:** Test applications for histogram (2D/3D), GMM, and compressor classes.
- **hipifly:** Preprocessor compatibility layer that maps CUDA API calls to HIP equivalents.
- **GPUWeightedGMMConfig.cmake:** CMake package config; consumed by `find_package(GPUWeightedGMM)`.

## Requirements

- **CMake:** minimum version 3.21
- **C++:** minimum C++17
- **GPU Backend:** CUDA (≥ compute capability 7.5 recommended) or HIP/ROCm (gfx90a default).
- **Dependencies:** None beyond the GPU platform SDK.

## Building and Running Tests

```bash
cd test
mkdir build && cd build
```

### CUDA

```bash
cmake ..
make
ctest
```

### HIP/ROCm

```bash
cmake .. -DHIP_ON=ON
make
ctest
```

## Integrating into Your Project

The library exposes a single CMake interface target `gpu_weighted_gmm` via `find_package`.
All include paths, the `HIPIFLY` compile definition (HIP builds), and HIP link flags propagate
automatically to any target that links against it.

### Minimal CMakeLists.txt

```cmake
cmake_minimum_required(VERSION 3.21)
project(MyProject LANGUAGES CXX)

option(HIP_ON "Use HIP backend instead of CUDA" OFF)
if(HIP_ON)
    find_package(HIP REQUIRED)
    enable_language(HIP)
else()
    enable_language(CUDA)
endif()

# Default: looks for GPU-Weighted-GMM next to your project.
# Override with: cmake -DGPUWeightedGMM_DIR=/path/to/GPU-Weighted-GMM ..
set(GPUWeightedGMM_DIR "${CMAKE_CURRENT_LIST_DIR}/../GPU-Weighted-GMM"
    CACHE PATH "Path to the GPU-Weighted-GMM source tree")
find_package(GPUWeightedGMM REQUIRED)

add_executable(my_app src/my_app.cu)
if(HIP_ON)
    set_source_files_properties(src/my_app.cu PROPERTIES LANGUAGE HIP)
else()
    set_source_files_properties(src/my_app.cu PROPERTIES LANGUAGE CUDA)
endif()
target_link_libraries(my_app PRIVATE gpu_weighted_gmm)
```

A complete working example is provided in the `gmmExample/` directory.

## Configuration

Compile-time parameters are set in two header files — edit them before building:

**`histogram/histogramConfig.cuh`** — histogram resolution and velocity range:
```cpp
namespace particleHistogram::config {
    inline constexpr int  PARTICLE_HISTOGRAM2D_RES_1 = 100; // bins in dim 1 (2D)
    inline constexpr int  PARTICLE_HISTOGRAM2D_RES_2 = 100; // bins in dim 2 (2D)
    inline constexpr int  PARTICLE_HISTOGRAM3D_RES_1 = 100; // bins in dim 1 (3D)
    // ... (3D dim 2, dim 3 analogous)
    inline constexpr bool             HISTOGRAM_FIXED_RANGE    = true;
    inline constexpr cudaCommonType   MIN_VELOCITY_HIST_E      = -1.6;
    inline constexpr cudaCommonType   MAX_VELOCITY_HIST_E      =  1.6;
}
```

**`weighted-GMM/GMMConfig.cuh`** — GMM solver settings:
```cpp
namespace weightedGMM::config {
    inline constexpr bool           NORMALIZE_DATA_FOR_GMM = false;
    inline constexpr bool           CHECK_COVMATRIX_GMM    = true;
    inline constexpr cudaCommonType EPS_COVMATRIX_GMM      = 1e-4;
    inline constexpr bool           PRUNE_COMPONENTS_GMM   = false;
    inline constexpr cudaCommonType PRUNE_THRESHOLD_GMM    = 0.005;
}
```

## Usage

1. Point CMake at the library root with `-DGPUWeightedGMM_DIR=...` or place your project next to the `GPU-Weighted-GMM` folder (default).
2. Call `find_package(GPUWeightedGMM REQUIRED)` and `target_link_libraries(... PRIVATE gpu_weighted_gmm)`.
3. Set compile-time options in `histogramConfig.cuh` and `GMMConfig.cuh` before building.
4. Include the single top-level header for the component you need:
   - Histogram only: `#include "histogram.cuh"`
   - GMM only: `#include "cudaGMM.cuh"`
   - Full pipeline: `#include "histGMMCompressor.cuh"`
5. Construct `ParticleHistogram<DIM>` or `HistGMMCompressor<T, DIM, useWeights, WeightT>` with the desired dimensionality; histogram size is inferred from the configuration at compile time.
6. Optionally tune GMM initialization by passing a `histogramGMMCompressor::GMMInitialParameters<T>` to the `HistGMMCompressor` constructor or to `setGMMInitialParameters(...)`. Without one, the built-in default initialization is used.

## Citation

If you use this library in your work, please cite:

> Hu, A., Pennati, L., Peng, I., Markidis, S. (2025). *Physics-Aware Compression of Plasma Distribution Functions with GPU-Accelerated Gaussian Mixture Models*. In: Lees, M.H., et al. Computational Science – ICCS 2025. ICCS 2025. Lecture Notes in Computer Science, vol 15905. Springer, Cham. [https://doi.org/10.1007/978-3-031-97632-2\_3](https://doi.org/10.1007/978-3-031-97632-2_3)


