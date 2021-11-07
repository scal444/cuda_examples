// Cuda implementation of capped_distance

#include <cstdio>
#include <vector>

#include "capped_distance.h"

#define checkReturn(ans) gpuAssert((ans), __FILE__, __LINE__)
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

__global__ void distanceMatrix(const double* d1, size_t d1_size, const double* d2, size_t d2_size, double* result) {
    size_t d1_idx = threadIdx.x + blockIdx.x * blockDim.x;
    size_t d2_idx = threadIdx.y + blockIdx.y * blockDim.y;

    if (d1_idx >= d1_size || d2_idx  >= d2_size) {
        return;
    }

    // Need to account for 3 floats in an element
    double dx = d1[3 * d1_idx] - d2[3 * d2_idx];
    double dy = d1[3 * d1_idx + 1] - d2[3 * d2_idx + 1];
    double dz = d1[3 * d1_idx + 2] - d2[3 * d2_idx + 2];


    size_t result_idx = d1_idx * d2_size + d2_idx;
    result[result_idx] = sqrt(dx * dx + dy * dy + dz * dz);
}

Distances CappedDistanceCuda(const std::vector<DVec>& d1, const std::vector<DVec>& d2, double cap) {
    Distances output;
    if (d1.empty() || d2.empty()) {
        return output;
    }

    // Reinterpret as contiguous doubles and compute number of bytes
    const double* d1_ptr = &d1[0][0];
    const double* d2_ptr = &d2[0][0];
    size_t d1_size = d1.size() * 3 * sizeof(double);
    size_t d2_size = d2.size() * 3 * sizeof(double);

    // Allocate and copy over to the GPU
    double* d1_dev_ptr = nullptr;
    double* d2_dev_ptr = nullptr;
    double* multiplication_matrix_dev_ptr = nullptr;
    size_t multiplication_matrix_size = d1.size() * d2.size() * sizeof(double);

    checkReturn(cudaMalloc(&d1_dev_ptr, d1_size));
    checkReturn(cudaMalloc(&d2_dev_ptr, d2_size));
    checkReturn(cudaMalloc(&multiplication_matrix_dev_ptr, multiplication_matrix_size));

    checkReturn(cudaMemcpy(d1_dev_ptr, d1_ptr, d1_size, cudaMemcpyHostToDevice));
    checkReturn(cudaMemcpy(d2_dev_ptr, d2_ptr, d2_size, cudaMemcpyHostToDevice));

    // debuggy thign, remove
    cudaMemset(multiplication_matrix_dev_ptr, -2.0, multiplication_matrix_size);



    // Execute kernel
    constexpr size_t blockStride = 16;
    const size_t numBlocks_d1 = d1.size() % blockStride == 0 ? d1.size() / blockStride : d1.size() / blockStride + 1;
    const size_t numBlocks_d2 = d2.size() % blockStride == 0 ? d2.size() / blockStride : d2.size() / blockStride + 1;
    const dim3 gridDimensions(numBlocks_d1, numBlocks_d2);
    const dim3 blockDimensions(16, 16);


    distanceMatrix<<<gridDimensions,blockDimensions>>>(d1_dev_ptr, d1.size(), d2_dev_ptr, d2.size(), multiplication_matrix_dev_ptr);
    // Copy results back over
    std::vector<double> results(d1.size() * d2.size(), -1);
    checkReturn(cudaMemcpy(results.data(), multiplication_matrix_dev_ptr, multiplication_matrix_size, cudaMemcpyDeviceToHost));

    for (size_t i = 0; i < d1.size(); i++) {
        for (size_t j = 0; j < d2.size(); j++) {
            size_t index = i * d2.size() + j;
            if (double d = results[index]; d < cap) {
                output.idx1.push_back(i);
                output.idx2.push_back(j);
                output.distances.push_back(d);
            }
        }
    }

    checkReturn(cudaFree(d1_dev_ptr));
    checkReturn(cudaFree(d2_dev_ptr));
    checkReturn(cudaFree(multiplication_matrix_dev_ptr));
    return output;
}
