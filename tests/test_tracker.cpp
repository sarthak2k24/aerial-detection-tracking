#include <gtest/gtest.h>

#include "Tracker.h"

namespace {
Detection car(float x, float y) { return {{x, y, 20, 10}, 2, 0.9f}; }
}  // namespace

TEST(Tracker, ConfirmsTrackAfterMinHits) {
    Tracker tracker;  // minHits = 3
    EXPECT_TRUE(tracker.update({car(0, 0)}).empty());
    EXPECT_TRUE(tracker.update({car(1, 0)}).empty());
    auto tracks = tracker.update({car(2, 0)});
    ASSERT_EQ(tracks.size(), 1u);
    EXPECT_EQ(tracks[0].id, 1);
}

TEST(Tracker, KeepsSameIdAsObjectMoves) {
    Tracker tracker;
    std::vector<Track> tracks;
    for (int i = 0; i < 10; ++i) tracks = tracker.update({car(i * 2.0f, 0)});
    ASSERT_EQ(tracks.size(), 1u);
    EXPECT_EQ(tracks[0].id, 1);
    EXPECT_FLOAT_EQ(tracks[0].box.x, 18.0f);
}

TEST(Tracker, SeparateObjectsGetSeparateIds) {
    Tracker tracker;
    std::vector<Track> tracks;
    for (int i = 0; i < 3; ++i) tracks = tracker.update({car(0, 0), car(200, 200)});
    ASSERT_EQ(tracks.size(), 2u);
    EXPECT_NE(tracks[0].id, tracks[1].id);
}

TEST(Tracker, DoesNotMatchAcrossClasses) {
    Tracker tracker;
    tracker.update({car(0, 0)});
    Detection person{{0, 0, 20, 10}, 0, 0.9f};
    tracker.update({person});
    EXPECT_EQ(tracker.allTracks().size(), 2u);
}

TEST(Tracker, SurvivesShortOcclusion) {
    Tracker tracker;
    for (int i = 0; i < 3; ++i) tracker.update({car(0, 0)});
    for (int i = 0; i < 5; ++i) tracker.update({});   // object hidden
    auto tracks = tracker.update({car(0, 0)});
    ASSERT_EQ(tracks.size(), 1u);
    EXPECT_EQ(tracks[0].id, 1);  // same ID after occlusion
}

TEST(Tracker, DropsTrackAfterMaxMisses) {
    Tracker tracker(Tracker::Config{0.3f, 2, 1});
    tracker.update({car(0, 0)});
    for (int i = 0; i < 3; ++i) tracker.update({});
    EXPECT_TRUE(tracker.allTracks().empty());
}
