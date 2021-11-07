#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include "capped_distance.h"

using ::testing::ElementsAre;
using ::testing::SizeIs;
using ::testing::DoubleNear;

// Demonstrate some basic assertions.
TEST(CappedDistanceTest, EmptyInput) {
    std::vector<DVec> a, b;
    Distances d = CappedDistance(a, b, 0);
    EXPECT_THAT(d.idx1, SizeIs(0));
    EXPECT_THAT(d.idx2, SizeIs(0));
    EXPECT_THAT(d.distances, SizeIs(0));
}

TEST(CappedDistanceTest, NoMatchInCutoff) {
    std::vector<DVec> a = {{0.1, 0.1, 0.1}};
    std::vector<DVec> b = {{0.95, 0.1, 0.1}};
    Distances d = CappedDistance(a, b, 0.2);
    EXPECT_THAT(d.idx1, SizeIs(0));
    EXPECT_THAT(d.idx2, SizeIs(0));
    EXPECT_THAT(d.distances, SizeIs(0));
}

TEST(CappedDistanceTest, Matches) {
    std::vector<DVec> a = {{0.0, 0.0, 0.0}, {0.1, 0.1, 0.1}};
    std::vector<DVec> b = {{-0.1, 0, 0}, {0.1, 0.2, 0.1}, {0.05, 0.1, 0.17}};
    Distances d = CappedDistance(a, b, 0.15);
    EXPECT_THAT(d.idx1, ElementsAre(0,1, 1));
    EXPECT_THAT(d.idx2, ElementsAre(0,1,2));
    EXPECT_THAT(d.distances, ElementsAre(DoubleNear(0.1, 0.001), DoubleNear(0.1, 0.001), DoubleNear(0.086, 0.001)));
}

TEST(CappedDistanceTest, CudaEmptyInput) {
    std::vector<DVec> a, b;
    Distances d = CappedDistanceCuda(a, b, 0);
    EXPECT_THAT(d.idx1, SizeIs(0));
    EXPECT_THAT(d.idx2, SizeIs(0));
    EXPECT_THAT(d.distances, SizeIs(0));
}

TEST(CappedDistanceTest, CudaNoMatchInCutoff) {
    std::vector<DVec> a = {{0.1, 0.1, 0.1}};
    std::vector<DVec> b = {{0.95, 0.1, 0.1}};
    Distances d = CappedDistanceCuda(a, b, 0.2);
    EXPECT_THAT(d.idx1, SizeIs(0));
    EXPECT_THAT(d.idx2, SizeIs(0));
    EXPECT_THAT(d.distances, SizeIs(0));
}
TEST(CappedDistanceTest, CudaMatches) {
    std::vector<DVec> a = {{0.0, 0.0, 0.0}, {0.1, 0.1, 0.1}};
    std::vector<DVec> b = {{-0.1, 0, 0}, {0.1, 0.2, 0.1}, {0.05, 0.1, 0.17}};
    Distances d = CappedDistanceCuda(a, b, 0.15);
    EXPECT_THAT(d.idx1, ElementsAre(0,1, 1));
    EXPECT_THAT(d.idx2, ElementsAre(0,1,2));
    EXPECT_THAT(d.distances, ElementsAre(DoubleNear(0.1, 0.001), DoubleNear(0.1, 0.001), DoubleNear(0.086, 0.001)));
}
