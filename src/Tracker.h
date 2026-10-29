#pragma once

#include <vector>

#include "Geometry.h"

struct Track {
    int id = -1;
    Box box;
    int classId = -1;
    int hits = 0;     // frames this track has been matched
    int misses = 0;   // consecutive frames without a match
};

// Multi-object tracker that keeps a stable ID per object across frames.
//
// Current version: greedy IoU matching (highest-overlap pairs first).
// Planned: Kalman-filter motion prediction + Hungarian assignment (SORT).
class Tracker {
public:
    struct Config {
        float minIou = 0.3f;   // minimum overlap to match a detection to a track
        int maxMisses = 15;    // drop a track after this many unmatched frames
        int minHits = 3;       // frames before a track is reported as confirmed
    };

    Tracker() = default;
    explicit Tracker(Config cfg) : cfg_(cfg) {}

    // Updates tracks with this frame's detections and returns confirmed tracks.
    std::vector<Track> update(const std::vector<Detection>& detections);

    const std::vector<Track>& allTracks() const { return tracks_; }

private:
    Config cfg_;
    std::vector<Track> tracks_;
    int nextId_ = 1;
};
