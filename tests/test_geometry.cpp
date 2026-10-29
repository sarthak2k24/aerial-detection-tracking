#include <gtest/gtest.h>

#include "Geometry.h"

TEST(Iou, IdenticalBoxesIsOne) {
    Box a{10, 10, 20, 20};
    EXPECT_FLOAT_EQ(iou(a, a), 1.0f);
}

TEST(Iou, DisjointBoxesIsZero) {
    EXPECT_FLOAT_EQ(iou({0, 0, 10, 10}, {50, 50, 10, 10}), 0.0f);
}

TEST(Iou, HalfOverlap) {
    // Overlap 5x10 = 50, union = 100 + 100 - 50 = 150
    EXPECT_NEAR(iou({0, 0, 10, 10}, {5, 0, 10, 10}), 50.0f / 150.0f, 1e-6);
}

TEST(Iou, ZeroAreaBoxIsZero) {
    EXPECT_FLOAT_EQ(iou({0, 0, 0, 0}, {0, 0, 0, 0}), 0.0f);
}

TEST(Nms, SuppressesOverlappingSameClass) {
    std::vector<Detection> dets = {
        {{0, 0, 10, 10}, 0, 0.9f},
        {{1, 1, 10, 10}, 0, 0.8f},   // heavy overlap with first -> dropped
        {{50, 50, 10, 10}, 0, 0.7f}, // far away -> kept
    };
    auto kept = nms(dets, 0.5f);
    ASSERT_EQ(kept.size(), 2u);
    EXPECT_FLOAT_EQ(kept[0].score, 0.9f);
    EXPECT_FLOAT_EQ(kept[1].score, 0.7f);
}

TEST(Nms, KeepsOverlappingDifferentClasses) {
    std::vector<Detection> dets = {
        {{0, 0, 10, 10}, 0, 0.9f},
        {{0, 0, 10, 10}, 2, 0.8f},
    };
    EXPECT_EQ(nms(dets, 0.5f).size(), 2u);
}

TEST(Nms, ReturnsSortedByScore) {
    std::vector<Detection> dets = {
        {{0, 0, 10, 10}, 0, 0.3f},
        {{100, 0, 10, 10}, 0, 0.9f},
    };
    auto kept = nms(dets, 0.5f);
    ASSERT_EQ(kept.size(), 2u);
    EXPECT_FLOAT_EQ(kept[0].score, 0.9f);
}
