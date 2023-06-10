#include <atomic>
#include <opencv2/opencv.hpp>
#include <thread>

#define STRIP_FLAG_HELP 1
#include <gflags/gflags.h>

#include "sam.h"

DEFINE_string(pre_model, "models/sam_preprocess.onnx", "Path to the preprocessing model");
DEFINE_string(sam_model, "models/sam_vit_h_4b8939.onnx", "Path to the sam model");
DEFINE_string(image, "images/input.jpg", "Path to the image to segment");
DEFINE_string(pre_device, "cpu", "cpu or cuda:0(1,2,3...)");
DEFINE_string(sam_device, "cpu", "cpu or cuda:0(1,2,3...)");
DEFINE_bool(h, false, "Show help");

bool parseDeviceName(const std::string& name, Sam::Parameter::Provider& provider) {
  if (name == "cpu") {
    provider.deviceType = 0;
    return true;
  }
  if (name.substr(0, 5) == "cuda:") {
    provider.deviceType = 1;
    provider.gpuDeviceId = std::stoi(name.substr(5));
    return true;
  }
  return false;
}

int main(int argc, char** argv) {
  gflags::ParseCommandLineNonHelpFlags(&argc, &argv, true);
  if (FLAGS_h) {
    std::cout << "Example: ./sam_cpp_test -pre_model=\"models/sam_preprocess.onnx\" "
                 "-sam_model=\"models/sam_vit_h_4b8939.onnx\" "
                 "-image=\"images/input.jpg\" -pre_device=\"cpu\" -sam_device=\"cpu\""
              << std::endl;
    return 0;
  }

  std::cout << "Preprocess device: " << FLAGS_pre_device << "; Sam device: " << FLAGS_sam_device
            << std::endl;

  Sam::Parameter param(FLAGS_pre_model, FLAGS_sam_model, std::thread::hardware_concurrency());
  if (!parseDeviceName(FLAGS_pre_device, param.providers[0]) ||
      !parseDeviceName(FLAGS_sam_device, param.providers[1])) {
    std::cerr << "Unable to parse device name" << std::endl;
  }

  std::cout << "Loading model..." << std::endl;
  Sam sam(param);  // FLAGS_pre_model, FLAGS_sam_model, std::thread::hardware_concurrency());

  auto inputSize = sam.getInputSize();
  if (inputSize.empty()) {
    std::cout << "Sam initialization failed" << std::endl;
    return -1;
  }

  cv::Mat image = cv::imread(FLAGS_image, -1);
  if (image.empty()) {
    std::cout << "Image loading failed" << std::endl;
    return -1;
  }
  std::cout << "Resize image to " << inputSize << std::endl;
  cv::resize(image, image, inputSize);
  std::cout << "Loading image..." << std::endl;
  if (!sam.loadImage(image)) {
    std::cout << "Image loading failed" << std::endl;
    return -1;
  }

  std::cout << "Now click on the image (press q/esc to quit; press c to clear selection; press a "
               "to run automatic segmentation)\n"
            << "Ctrl+Left click to select foreground, Ctrl+Right click to select background\n";

  std::list<cv::Point3i> clickedPoints;
  cv::Point3i newClickedPoint(-1, 0, 0);
  cv::Mat outImage = image.clone();

  auto g_windowName = "Segment Anything CPP Demo";
  cv::namedWindow(g_windowName, 0);
  cv::setMouseCallback(
      g_windowName,
      [](int event, int x, int y, int flags, void* userdata) {
        int code = -1;
        if (event == cv::EVENT_LBUTTONDOWN) {
          code = 2;
        } else if (event == cv::EVENT_RBUTTONDOWN) {
          code = 0;
        }
        if (code >= 0) {
          if ((flags & cv::EVENT_FLAG_CTRLKEY) == cv::EVENT_FLAG_CTRLKEY) {
            // If ctrl is pressed, then append it to the list later
            code += 1;
          }
          *(cv::Point3i*)userdata = {x, y, code};
        }
      },
      &newClickedPoint);

#define SHOW_TIME                                                     \
  std::cout << "Time elapsed: "                                       \
            << std::chrono::duration_cast<std::chrono::milliseconds>( \
                   std::chrono::system_clock::now() - timeNow)        \
                   .count()                                           \
            << " ms" << std::endl;

  bool bRunning = true;
  while (bRunning) {
    const auto timeNow = std::chrono::system_clock::now();

    if (newClickedPoint.x > 0) {
      std::list<cv::Point> points, nagativePoints;
      if (newClickedPoint.z % 2 == 0) {
        clickedPoints = {newClickedPoint};
      } else {
        clickedPoints.push_back(newClickedPoint);
      }

      for (auto& p : clickedPoints) {
        if (p.z >= 2) {
          points.push_back({p.x, p.y});
        } else {
          nagativePoints.push_back({p.x, p.y});
        }
      }
      cv::Mat mask = sam.getMask(points, nagativePoints);
      newClickedPoint.x = -1;
      SHOW_TIME

      // apply mask to image
      outImage = cv::Mat::zeros(image.size(), CV_8UC3);
      for (int i = 0; i < image.rows; i++) {
        for (int j = 0; j < image.cols; j++) {
          auto bFront = mask.at<uchar>(i, j) > 0;
          float factor = bFront ? 1.0 : 0.2;
          outImage.at<cv::Vec3b>(i, j) = image.at<cv::Vec3b>(i, j) * factor;
        }
      }

      for (auto& p : points) {
        cv::circle(outImage, p, 2, {0, 255, 255}, -1);
      }
      for (auto& p : nagativePoints) {
        cv::circle(outImage, p, 2, {255, 0, 0}, -1);
      }
    } else if (newClickedPoint.x == -2) {
      newClickedPoint.x = -1;
      int step = 40;
      cv::Size sampleSize = {image.cols / step, image.rows / step};

      std::cout << "Automatically generating masks with " << sampleSize.area()
                << " input points ..." << std::endl;

      auto mask = sam.autoSegment(
          sampleSize, [](double v) { std::cout << "\rProgress: " << int(v * 100) << "%\t"; });
      SHOW_TIME

      const double overlayFactor = 0.5;
      const int maxMaskValue = 255 * (1 - overlayFactor);
      outImage = cv::Mat::zeros(image.size(), CV_8UC3);

      static std::map<int, cv::Vec3b> colors;

      for (int i = 0; i < image.rows; i++) {
        for (int j = 0; j < image.cols; j++) {
          auto value = (int)mask.at<double>(i, j);
          if (value <= 0) {
            continue;
          }

          auto it = colors.find(value);
          if (it == colors.end()) {
            colors.insert(it, {value, cv::Vec3b(rand() % maxMaskValue, rand() % maxMaskValue,
                                                rand() % maxMaskValue)});
          }

          outImage.at<cv::Vec3b>(i, j) = it->second + image.at<cv::Vec3b>(i, j) * overlayFactor;
        }
      }

      // draw circles on the image to indicate the sample points
      for (int i = 0; i < sampleSize.height; i++) {
        for (int j = 0; j < sampleSize.width; j++) {
          cv::circle(outImage, {j * step, i * step}, 2, {0, 0, 255}, -1);
        }
      }
    }

    cv::imshow(g_windowName, outImage);
    int key = cv::waitKeyEx(100);
    switch (key) {
      case 27:
      case 'Q':
      case 'q': {
        bRunning = false;
      } break;
      case 'C':
      case 'c': {
        clickedPoints.clear();
        newClickedPoint.x = -1;
        outImage = image.clone();
      } break;
      case 'A':
      case 'a': {
        clickedPoints.clear();
        newClickedPoint.x = -2;
        outImage = image.clone();
      }
    }
  }

  cv::destroyWindow(g_windowName);

  return 0;
}
