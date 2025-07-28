# Histogram-GMM Data Compression Scheme

A C++ library implementing histogramming and Gaussian Mixture Model (GMM) data compression for particle data on accelerators, with support for both CUDA and HIP/ROCm backends.

## Features

- **Histogram Stage:** Efficient GPU-based histogramming of particle data.
- **GMM Stage:** Weighted Gaussian Mixture Model fitting for data compression.
- **Full Pipeline:** Combined histogram + GMM compressor for maximum compression.
- **Modular:** Use either histogram-only, GMM-only, or full pipeline.
- **Backend Support:** Compatible with CUDA and HIP/ROCm.

> **Note**: GMM initialization critically affects convergence and should be tailored to your specific data.

## Folder Structure

- **common:** Shared routines, such as reduction kernels.
- **histogram:** Defines the histogram class.
- **weighted-GMM:** Defines the weighted-GMM class.
- **histogram-GMM-compressor:** Implements the full histogram + GMM compression pipeline.
- **test:** Example applications demonstrating usage of the histogram, GMM, and compressor classes.
- **hipifly:** Defines preprocessor macros to convert the CUDA API calls to HIP calls.


## Requirements
- **CMAKE:** minimum version 3.21
- **c++:** minimum 17
- **GPU Backend:** CUDA or HIP/ROCm support required.
- **Dependencies:** None beyond the GPU platform SDK.

## Building and Compiling Tests

```bash
cd test
mkdir build && cd build
```

### CUDA

```bash
cmake ..
make
```

### HIP/ROCm

```bash
cmake .. -DHIP_ON=ON
make
```

## Usage

1. Copy the library folders (`common`, `hipifly`, `histogram`, `weighted-GMM`, `histogram-GMM-compressor`) into your code source directory.
2. Add the general settings and `HIP_ON` option in your `CMakeLists.txt` referencing the `CMakeLists.txt` example in the `test`.  Add the necessary `add_subdirectory` or include paths in your `CMakeLists.txt`, referencing the examples in the `test` folder. 
3. Configure and build using CMake as shown above.

## Citation

If you use this library in your work, please cite:

> Hu, A., Pennati, L., Peng, I., Markidis, S. (2025). *Physics-Aware Compression of Plasma Distribution Functions with GPU-Accelerated Gaussian Mixture Models*. In: Lees, M.H., et al. Computational Science – ICCS 2025. ICCS 2025. Lecture Notes in Computer Science, vol 15905. Springer, Cham. [https://doi.org/10.1007/978-3-031-97632-2\_3](https://doi.org/10.1007/978-3-031-97632-2_3)

