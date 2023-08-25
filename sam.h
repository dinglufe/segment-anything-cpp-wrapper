#ifndef SAMCPP__SAM_H_
#define SAMCPP__SAM_H_

#include <opencv2/core.hpp>
#include <string>
#include <list>

struct SamModel;

#if _MSC_VER
class __declspec(dllexport) Sam {
#else
class Sam {
#endif
  SamModel* m_model{nullptr};

 public:
  struct Parameter {
    // Partial options of OrtCUDAProviderOptions to hide the dependency on onnxruntime
    struct Provider {
      // deviceType: 0 - CPU, 1 - CUDA
      int gpuDeviceId{0}, deviceType{0};
      size_t gpuMemoryLimit{0};
    };
    Provider providers[2];  // 0 - embedding, 1 - segmentation
    std::string models[2];  // 0 - embedding, 1 - segmentation
    int threadsNumber{1};
    Parameter(const std::string& preModelPath, const std::string& samModelPath, int threadsNumber) {
      models[0] = preModelPath;
      models[1] = samModelPath;
      this->threadsNumber = threadsNumber;
    }
  };

  // This constructor is deprecated (cpu runtime only)
  Sam(const std::string& preModelPath, const std::string& samModelPath, int threadsNumber);
  // Recommended constructor
  Sam(const Parameter& param);
  ~Sam();

  cv::Size getInputSize() const;
  bool loadImage(const cv::Mat& image);

  cv::Mat getMask(const std::list<cv::Point>& points, const std::list<cv::Point>& negativePoints,
                  const cv::Rect& roi, double* iou = nullptr) const;
  cv::Mat getMask(const std::list<cv::Point>& points, const std::list<cv::Point>& negativePoints,
                  double* iou = nullptr) const;
  cv::Mat getMask(const cv::Point& point, double* iou = nullptr) const;

  using cbProgress = void (*)(double);
  cv::Mat autoSegment(const cv::Size& numPoints, cbProgress cb = {},
                      const double iouThreshold = 0.86, const double minArea = 100,
                      int* numObjects = nullptr) const;
};

#endif  // SAMCPP__SAM_H_
