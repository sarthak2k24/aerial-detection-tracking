#pragma once

#include <vector>

// Axis-aligned bounding box in pixel coordinates.
struct Box {
    float x = 0, y = 0, w = 0, h = 0;
    float area() const { return w > 0 && h > 0 ? w * h : 0.0f; }
    float cx() const { return x + w / 2; }
    float cy() const { return y + h / 2; }
};

struct Detection {
    Box box;
    int classId = -1;
    float score = 0.0f;
};

// Intersection-over-union of two boxes, in [0, 1].
float iou(const Box& a, const Box& b);

// Class-aware non-maximum suppression. Keeps the highest-scoring box and drops
// any same-class box overlapping it by more than iouThreshold.
std::vector<Detection> nms(std::vector<Detection> dets, float iouThreshold);
