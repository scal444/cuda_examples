#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include "capped_distance.h"

using ::testing::ContainerEq;
using ::testing::DoubleNear;
using ::testing::Each;
using ::testing::ElementsAre;
using ::testing::SizeIs;

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

TEST(CappedDistanceTest, LargeScaleCorrect) {
    DVec d1 = {0.1, 0.2, 0.3};
    DVec d2 = {0.0, 0.2, 0.4};
    double want_val = 0.14142135623;

    std::vector<DVec> v1(1000, d1);
    std::vector<DVec> v2(500, d2);

    Distances d = CappedDistanceCuda(v1, v2, 0.15);

    // Want 0, 0, 0, .... 1, 1, 1,...
    std::vector<size_t> want_idx1;
    std::vector<size_t> want_idx2;

    ASSERT_EQ(d.idx1.size(), v1.size() * v2.size());
    EXPECT_THAT(d.distances, Each(DoubleNear(want_val, 0.001)));
    for (int i = 0; i < v1.size(); i++) {
        for (int j = 0; j < v2.size(); j++) {
            int idx = i * v2.size() + j;
            ASSERT_EQ(d.idx1[idx], i);
            ASSERT_EQ(d.idx2[idx], j);
        }
    }
}
