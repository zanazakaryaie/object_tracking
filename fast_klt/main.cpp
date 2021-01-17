#include <iostream>
#include <opencv2/opencv.hpp>
#include "FastKltTracker.hpp"

using namespace std;
using namespace cv;
using namespace cv::dnn;


int main()
{
    cv::VideoCapture cap("../../Haddock.mp4");

    const std::string caffeConfigFile = "../../deploy.prototxt";
    const std::string caffeWeightFile = "../../res10_300x300_ssd_iter_140000_fp16.caffemodel";
    Net net = cv::dnn::readNetFromCaffe(caffeConfigFile, caffeWeightFile);
    net.setPreferableBackend(DNN_BACKEND_DEFAULT);

    const size_t inWidth = 300;
    const size_t inHeight = 300;
    const double inScaleFactor = 1.0;
    const float confidenceThreshold = 0.7;
    const cv::Scalar meanVal(104.0, 177.0, 123.0);


    Mat frame;
    bool do_detection = true;

    FastKltTracker tracker;

    FileStorage fs("../params.yaml", FileStorage::READ);

    int maxFeatures = fs["maxFeatures"];
    int fastThreshold = fs["fastThreshold"];
    int nRows = fs["nRows"];
    int nCols = fs["nCols"];
    int kltWindowSize = fs["kltWindowSize"];
    float shrinkRatio = fs["shrinkRatio"];
    float ransacThreshold = fs["ransacThreshold"];

    fs.release();

    tracker.setParams(maxFeatures, cv::Size(nRows, nCols), fastThreshold, shrinkRatio, cv::Size(kltWindowSize, kltWindowSize), ransacThreshold);

    while (cap.isOpened())
    {
        cap.read(frame);

        if (frame.empty())
            break;

        int frameHeight = frame.rows;
        int frameWidth = frame.cols;

        if (do_detection)
        {
            cv::Mat inputBlob = cv::dnn::blobFromImage(frame, inScaleFactor, cv::Size(inWidth, inHeight), meanVal, false, false);

            net.setInput(inputBlob, "data");
            cv::Mat detection = net.forward("detection_out");
            cv::Mat detectionMat(detection.size[2], detection.size[3], CV_32F, detection.ptr<float>());

            float confidence = detectionMat.at<float>(0, 2);

            if (confidence > 0.5)
            {
                int x1 = static_cast<int>(detectionMat.at<float>(0, 3) * frameWidth);
                int y1 = static_cast<int>(detectionMat.at<float>(0, 4) * frameHeight);
                int x2 = static_cast<int>(detectionMat.at<float>(0, 5) * frameWidth);
                int y2 = static_cast<int>(detectionMat.at<float>(0, 6) * frameHeight);

                tracker.init(frame, Rect(x1,y1,x2-x1,y2-y1));
                cv::rectangle(frame, Point(x1,y1), Point(x2,y2), Scalar(0,255,0), 3);

                do_detection = false;

            }
        }
        else
        {
            cv::Rect2f trackedObj;
            bool status = tracker.update(frame, trackedObj);

            if (status)
                rectangle(frame, trackedObj, Scalar(0,255,0), 3);
            else
                do_detection = true;
        }

        imshow("Frmae", frame);

        char key = waitKey(1);

        if (key == 'q')
            break;
    }

    cap.release();

    return 0;
}
