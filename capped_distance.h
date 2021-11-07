#ifndef CUDA_EXAMPLES_CAPPED_DISTANCE_H
#define CUDA_EXAMPLES_CAPPED_DISTANCE_H

#include <cstddef>

#include <array>

typedef std::array<double, 3> DVec;

struct Distances {
    std::vector<size_t> idx1;
    std::vector<size_t> idx2;
    std::vector<double> distances;
};

enum class Backend {
    cpp = 0,
    cuda
};

Distances CappedDistance(const std::vector<DVec>& d1, const std::vector<DVec>& d2, double cap);

Distances CappedDistanceCuda(const std::vector<DVec>& d1, const std::vector<DVec>& d2, double cap);

#endif //CUDA_EXAMPLES_CAPPED_DISTANCE_H
