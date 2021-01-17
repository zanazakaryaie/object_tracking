#include <iostream>
#include <opencv2/opencv.hpp>
#include <dlib/image_processing.h>
#include <dlib/opencv/cv_image.h>


using namespace std;
using namespace cv;
using namespace cv::dnn;
using namespace dlib;


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

    correlation_tracker tracker;

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

                array2d<rgb_pixel> dlib_img;
                assign_image(dlib_img, cv_image<bgr_pixel>(frame));

                tracker.start_track(dlib_img, drectangle(x1,y1,x2,y2));

                cv::rectangle(frame, Point(x1,y1), Point(x2,y2), Scalar(0,255,0), 3);

                do_detection = false;

            }
        }
        else
        {
            array2d<rgb_pixel> dlib_img;
            assign_image(dlib_img, cv_image<bgr_pixel>(frame));

            double res = tracker.update(dlib_img);

            if (res > 10)
            {
                drectangle pos = tracker.get_position();
                cv::rectangle(frame, cv::Point(pos.left(), pos.top()), cv::Point(pos.right(), pos.bottom()), Scalar(0,255,0), 3);
            }
            else
            {
                do_detection = true;
            }
        }


        imshow("Frmae", frame);

        char key = waitKey(1);

        if (key == 'q')
            break;
    }

    cap.release();

    return 0;
}
