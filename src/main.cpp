// Detects and tracks objects in a video file or camera stream.
// Usage: detect_track <model.onnx> <video_path | camera_index> [--no-display] [--out out.mp4]

#include <algorithm>
#include <cctype>
#include <chrono>
#include <iostream>
#include <string>

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include "Detector.h"
#include "Tracker.h"

namespace {

cv::Scalar colorForId(int id) {
    // Deterministic, well-spread colors per track id
    return cv::Scalar((id * 67) % 256, (id * 131) % 256, (id * 199) % 256);
}

bool isNumber(const std::string& s) {
    return !s.empty() && std::all_of(s.begin(), s.end(), ::isdigit);
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0]
                  << " <model.onnx> <video_path | camera_index> [--no-display] [--out out.mp4]\n";
        return 1;
    }
    const std::string modelPath = argv[1];
    const std::string source = argv[2];
    bool display = true;
    std::string outPath;
    for (int i = 3; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--no-display") display = false;
        else if (a == "--out" && i + 1 < argc) outPath = argv[++i];
    }

    cv::VideoCapture cap;
    if (isNumber(source)) cap.open(std::stoi(source));
    else cap.open(source);
    if (!cap.isOpened()) {
        std::cerr << "Could not open video source: " << source << "\n";
        return 1;
    }

    Detector detector(modelPath);
    Tracker tracker;
    const auto& names = cocoClassNames();
    cv::VideoWriter writer;

    cv::Mat frame;
    int frameCount = 0;
    double totalMs = 0;

    while (cap.read(frame)) {
        auto t0 = std::chrono::steady_clock::now();
        auto detections = detector.detect(frame);
        auto tracks = tracker.update(detections);
        double ms = std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - t0).count();
        totalMs += ms;
        ++frameCount;

        for (const auto& t : tracks) {
            cv::Rect r(cv::Point(int(t.box.x), int(t.box.y)),
                       cv::Size(int(t.box.w), int(t.box.h)));
            auto color = colorForId(t.id);
            cv::rectangle(frame, r, color, 2);
            std::string label = "#" + std::to_string(t.id) + " " +
                                (t.classId >= 0 && t.classId < int(names.size()) ? names[t.classId] : "?");
            cv::putText(frame, label, r.tl() + cv::Point(0, -5), cv::FONT_HERSHEY_SIMPLEX, 0.5,
                        color, 2);
        }
        cv::putText(frame, cv::format("%.1f FPS  tracks: %zu", 1000.0 / ms, tracks.size()),
                    {10, 25}, cv::FONT_HERSHEY_SIMPLEX, 0.7, {0, 255, 0}, 2);

        if (!outPath.empty()) {
            if (!writer.isOpened()) {
                double fps = cap.get(cv::CAP_PROP_FPS);
                writer.open(outPath, cv::VideoWriter::fourcc('m', 'p', '4', 'v'),
                            fps > 0 ? fps : 30, frame.size());
            }
            writer.write(frame);
        }
        if (display) {
            cv::imshow("Aerial Detection & Tracking", frame);
            if (cv::waitKey(1) == 27) break;  // Esc to quit
        }
    }

    if (frameCount > 0) {
        std::cout << "Processed " << frameCount << " frames, average "
                  << totalMs / frameCount << " ms/frame ("
                  << 1000.0 * frameCount / totalMs << " FPS)\n";
    }
    return 0;
}
