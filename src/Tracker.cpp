#include "Tracker.h"

#include <algorithm>
#include <tuple>

std::vector<Track> Tracker::update(const std::vector<Detection>& detections) {
    // Score every (track, detection) pair of the same class
    std::vector<std::tuple<float, std::size_t, std::size_t>> pairs;
    for (std::size_t t = 0; t < tracks_.size(); ++t) {
        for (std::size_t d = 0; d < detections.size(); ++d) {
            if (tracks_[t].classId != detections[d].classId) continue;
            float overlap = iou(tracks_[t].box, detections[d].box);
            if (overlap >= cfg_.minIou) pairs.emplace_back(overlap, t, d);
        }
    }
    // Greedy: best overlaps get matched first
    std::sort(pairs.begin(), pairs.end(),
              [](const auto& a, const auto& b) { return std::get<0>(a) > std::get<0>(b); });

    std::vector<bool> trackUsed(tracks_.size(), false);
    std::vector<bool> detUsed(detections.size(), false);
    for (const auto& [overlap, t, d] : pairs) {
        if (trackUsed[t] || detUsed[d]) continue;
        trackUsed[t] = detUsed[d] = true;
        tracks_[t].box = detections[d].box;
        tracks_[t].hits++;
        tracks_[t].misses = 0;
    }

    for (std::size_t t = 0; t < tracks_.size(); ++t) {
        if (!trackUsed[t]) tracks_[t].misses++;
    }

    // Unmatched detections start new tracks
    for (std::size_t d = 0; d < detections.size(); ++d) {
        if (detUsed[d]) continue;
        Track tr;
        tr.id = nextId_++;
        tr.box = detections[d].box;
        tr.classId = detections[d].classId;
        tr.hits = 1;
        tracks_.push_back(tr);
    }

    // Drop stale tracks
    tracks_.erase(std::remove_if(tracks_.begin(), tracks_.end(),
                                 [&](const Track& t) { return t.misses > cfg_.maxMisses; }),
                  tracks_.end());

    std::vector<Track> confirmed;
    for (const auto& t : tracks_) {
        if (t.hits >= cfg_.minHits && t.misses == 0) confirmed.push_back(t);
    }
    return confirmed;
}
