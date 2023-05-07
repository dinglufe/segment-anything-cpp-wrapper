#include <atomic>
#include <opencv2/opencv.hpp>
#include <thread>

#define STRIP_FLAG_HELP 1
#include <gflags/gflags.h>

#include "sam.h"

DEFINE_string(pre_model, "models/sam_preprocess.onnx", "Path to the preprocessing model");
DEFINE_string(sam_model, "models/sam_vit_h_4b8939.onnx", "Path to the sam model");
DEFINE_string(image, "images/input.jpg", "Path to the image to segment");
DEFINE_bool(h, false, "Show help");

int main(int argc, char** argv) {
  gflags::ParseCommandLineNonHelpFlags(&argc, &argv, true);
  if (FLAGS_h) {
    std::cout << "Example: ./sam_cpp_test -pre_model=\"models/sam_preprocess.onnx\" "
                 "-sam_model=\"models/sam_vit_h_4b8939.onnx\" "
                 "-image=\"images/input.jpg\""
              << std::endl;
    return 0;
  }

  std::cout << "Loading model..." << std::endl;
  Sam sam(FLAGS_pre_model, FLAGS_sam_model, std::thread::hardware_concurrency());

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

  std::cout << "Now click on the image (press q/esc to quit; press c to clear selection)"
            << std::endl;

  cv::Point clickedPoint;
  cv::Mat outImage = image.clone();

  auto g_windowName = "Segment Anything CPP Demo";
  cv::namedWindow(g_windowName, 0);
  cv::setMouseCallback(
      g_windowName,
      [](int event, int x, int y, int flags, void* userdata) {
        if (event == cv::EVENT_LBUTTONDOWN) {
          *(cv::Point*)userdata = {x, y};
        }
      },
      &clickedPoint);

  bool bRunning = true;
  while (bRunning) {
    if (clickedPoint.x > 0 && clickedPoint.y > 0) {
      cv::Mat mask = sam.getMask(clickedPoint);
      clickedPoint = {};

      // apply mask to image
      outImage = cv::Mat::zeros(image.size(), CV_8UC3);
      for (int i = 0; i < image.rows; i++) {
        for (int j = 0; j < image.cols; j++) {
          auto bFront = mask.at<uchar>(i, j) > 0;
          float factor = bFront ? 1.0 : 0.2;
          outImage.at<cv::Vec3b>(i, j) = image.at<cv::Vec3b>(i, j) * factor;
        }
      }
    }
    cv::imshow(g_windowName, outImage);
    int key = cv::waitKeyEx(100);
    switch (key) {
      case 27:
      case 'q': {
        bRunning = false;
      } break;
      case 'c': {
        clickedPoint = {};
        outImage = image.clone();
      } break;
    }
  }

  cv::destroyWindow(g_windowName);

  return 0;
}
