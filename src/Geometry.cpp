#include "Geometry.h"

#include <algorithm>

float iou(const Box& a, const Box& b) {
    float x1 = std::max(a.x, b.x);
    float y1 = std::max(a.y, b.y);
    float x2 = std::min(a.x + a.w, b.x + b.w);
    float y2 = std::min(a.y + a.h, b.y + b.h);
    float inter = std::max(0.0f, x2 - x1) * std::max(0.0f, y2 - y1);
    float uni = a.area() + b.area() - inter;
    return uni > 0 ? inter / uni : 0.0f;
}

std::vector<Detection> nms(std::vector<Detection> dets, float iouThreshold) {
    std::sort(dets.begin(), dets.end(),
              [](const Detection& a, const Detection& b) { return a.score > b.score; });

    std::vector<Detection> kept;
    std::vector<bool> suppressed(dets.size(), false);
    for (std::size_t i = 0; i < dets.size(); ++i) {
        if (suppressed[i]) continue;
        kept.push_back(dets[i]);
        for (std::size_t j = i + 1; j < dets.size(); ++j) {
            if (!suppressed[j] && dets[j].classId == dets[i].classId &&
                iou(dets[i].box, dets[j].box) > iouThreshold) {
                suppressed[j] = true;
            }
        }
    }
    return kept;
}
