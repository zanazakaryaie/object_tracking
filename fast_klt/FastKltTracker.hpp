#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>

class FastKltTracker
{
public:
    FastKltTracker(void);
    void setParams(int maxPts=300, cv::Size Grid=cv::Size(4,5), int FastThreshold=5, float shrinkRat=0.2, cv::Size KltWindowSize=cv::Size(11, 11), float RansacThreshold=0.5);
    void init(const cv::Mat& frame, const cv::Rect& object);
    bool update(const cv::Mat& frame, cv::Rect2f &trackedObj);

private:
    cv::Rect2f findIntersection(const cv::Rect2f& input);
    cv::Rect shrinkRect(const cv::Rect& input);
    cv::Rect2f boundingRect2f(const std::vector<cv::Point2f>& points);
    void keepStrongest( int N, std::vector<cv::KeyPoint>& keypoints);
    std::vector<cv::Point2f> detectGridFASTpoints(cv::Mat image, cv::Rect object);
    float track(const cv::Mat &img_1, const cv::Mat &img_2, std::vector<cv::Point2f>& points1, std::vector<cv::Point2f>& points2);

    int maxTotalKeypoints;
    int FastThresh;
    cv::Size grid;
    cv::Size KltWinSize;
    float RansacThresh;
    std::vector<cv::Point2f> prevCorners;
    std::vector<cv::Point2f> prevKeypoints;
    cv::Mat prevGrayFrame;
    float shrinkRatio;
};
