#pragma once

#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>

#include "Geometry.h"

// Runs a YOLOv8 model (exported to ONNX) with OpenCV's DNN module.
class Detector {
public:
    struct Config {
        int inputSize = 640;
        float scoreThreshold = 0.35f;
        float nmsThreshold = 0.45f;
    };

    explicit Detector(const std::string& onnxPath);
    Detector(const std::string& onnxPath, Config cfg);

    std::vector<Detection> detect(const cv::Mat& frame);

private:
    cv::dnn::Net net_;
    Config cfg_;
};

// COCO class names used by the default YOLOv8 weights.
const std::vector<std::string>& cocoClassNames();
