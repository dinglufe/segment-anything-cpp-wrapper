#ifndef SAMCPP__SAM_H_
#define SAMCPP__SAM_H_

#include <opencv2/core.hpp>
#include <string>

struct SamModel;

#if _MSC_VER
class __declspec(dllexport) Sam {
#else
class Sam {
#endif
  SamModel* m_model{nullptr};

 public:
  explicit Sam(const std::string& preModelPath, const std::string& samModelPath, int threadsNumber);
  ~Sam();

  cv::Size getInputSize() const;
  bool loadImage(const cv::Mat& image);

  cv::Mat getMask(const std::list<cv::Point>& points, const std::list<cv::Point>& negativePoints,
                  double* iou = nullptr) const;
  cv::Mat getMask(const cv::Point& point, double* iou = nullptr) const;

  using cbProgress = void (*)(double);
  cv::Mat autoSegment(const cv::Size& numPoints, cbProgress cb = {},
                      const double iouThreshold = 0.86, const double minArea = 100,
                      int* numObjects = nullptr) const;
};

#endif  // SAMCPP__SAM_H_
