#include "Detector.h"

#include <algorithm>

#include <opencv2/imgproc.hpp>

Detector::Detector(const std::string& onnxPath) : Detector(onnxPath, Config{}) {}

Detector::Detector(const std::string& onnxPath, Config cfg)
    : net_(cv::dnn::readNetFromONNX(onnxPath)), cfg_(cfg) {
    net_.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    net_.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
}

std::vector<Detection> Detector::detect(const cv::Mat& frame) {
    // Letterbox: pad to a square so the aspect ratio is preserved
    int side = std::max(frame.cols, frame.rows);
    cv::Mat square = cv::Mat::zeros(side, side, CV_8UC3);
    frame.copyTo(square(cv::Rect(0, 0, frame.cols, frame.rows)));

    cv::Mat blob = cv::dnn::blobFromImage(square, 1.0 / 255.0,
                                          cv::Size(cfg_.inputSize, cfg_.inputSize),
                                          cv::Scalar(), /*swapRB=*/true, /*crop=*/false);
    net_.setInput(blob);
    cv::Mat out = net_.forward();

    // YOLOv8 output: [1, 4 + numClasses, numAnchors] -> transpose to [numAnchors, 4 + numClasses]
    const int rows = out.size[2];
    const int dims = out.size[1];
    cv::Mat preds(dims, rows, CV_32F, out.ptr<float>());
    preds = preds.t();

    const float scale = static_cast<float>(side) / cfg_.inputSize;
    std::vector<Detection> dets;
    for (int i = 0; i < rows; ++i) {
        const float* row = preds.ptr<float>(i);
        cv::Mat scores(1, dims - 4, CV_32F, const_cast<float*>(row + 4));
        cv::Point classPt;
        double best;
        cv::minMaxLoc(scores, nullptr, &best, nullptr, &classPt);
        if (best < cfg_.scoreThreshold) continue;

        float cx = row[0], cy = row[1], w = row[2], h = row[3];
        Detection d;
        d.box = {(cx - w / 2) * scale, (cy - h / 2) * scale, w * scale, h * scale};
        d.classId = classPt.x;
        d.score = static_cast<float>(best);
        dets.push_back(d);
    }
    return nms(std::move(dets), cfg_.nmsThreshold);
}

const std::vector<std::string>& cocoClassNames() {
    static const std::vector<std::string> names = {
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
        "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
        "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
        "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
        "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
        "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
        "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
        "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
        "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
        "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
        "toothbrush"};
    return names;
}
