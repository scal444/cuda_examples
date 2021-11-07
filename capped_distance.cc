#include <cmath>

#include <vector>

#include "capped_distance.h"

static double singleDistance(const DVec& a, const DVec& b) {
    double dx = a[0] - b[0];
    double dy = a[1] - b[1];
    double dz = a[2] - b[2];

    return sqrt(dx * dx + dy * dy + dz * dz);
}

Distances CappedDistance(const std::vector<DVec>& d1, const std::vector<DVec>& d2, double cap) {
    Distances output;
    for (int i = 0; i < d1.size(); i++) {
        for (int j = 0; j < d2.size(); j++) {
            double distq = singleDistance(d1[i], d2[j]);
            if (double dist = singleDistance(d1[i], d2[j]); dist <= cap) {
                output.idx1.push_back(i);
                output.idx2.push_back(j);
                output.distances.push_back(dist);
            }
        }
    }
    return output;
}