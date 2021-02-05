#include "FastKltTracker.hpp"


//----------------------------------------------------------------------------
//----------------------------------------------------------------------------
FastKltTracker::FastKltTracker(void)
{

}


//----------------------------------------------------------------------------
//----------------------------------------------------------------------------
void FastKltTracker::setParams(int maxPts, cv::Size Grid, int FastThreshold, float shrinkRat, cv::Size KltWindowSize, float RansacThreshold)
{
    maxTotalKeypoints = maxPts;
    grid = Grid;
    FastThresh = FastThreshold;
    shrinkRatio = shrinkRat;
    KltWinSize = KltWindowSize;
    RansacThresh = RansacThreshold;
}


//----------------------------------------------------------------------------
//----------------------------------------------------------------------------
void FastKltTracker::init(const cv::Mat& frame, const cv::Rect& object)
{
    if (frame.channels()==3)
        cvtColor(frame, prevGrayFrame, cv::COLOR_BGR2GRAY);
    else
        prevGrayFrame = frame;

    cv::Rect shrinkedObj = shrinkRect(object);
    prevKeypoints = detectGridFASTpoints(prevGrayFrame, shrinkedObj);

    float x = (float)object.x;
    float y = (float)object.y;
    float w = (float)object.width;
    float h = (float)object.height;

    prevCorners.resize(4);
    prevCorners[0] = cv::Point2f(x,y);
    prevCorners[1] = cv::Point2f(x+w,y);
    prevCorners[2] = cv::Point2f(x+w,y+h);
    prevCorners[3] = cv::Point2f(x,y+h);
}


//----------------------------------------------------------------------------
//----------------------------------------------------------------------------
bool FastKltTracker::update(const cv::Mat& frame, cv::Rect2f &output)
{
    if (prevKeypoints.size() < 10) //Few points detected
        return false;

    cv::Mat currFrameGray;
    if (frame.channels()==3)
        cvtColor(frame, currFrameGray, cv::COLOR_BGR2GRAY);
    else
        currFrameGray = frame;


    std::vector<cv::Point2f> currKeypoints = prevKeypoints;
    float prob = track(prevGrayFrame, currFrameGray, prevKeypoints, currKeypoints);

    if (prob < 0.6) //Few points tracked
        return false;

    cv::Mat inliers;
    cv::Mat geometry = cv::estimateAffinePartial2D(prevKeypoints, currKeypoints, inliers, cv::RANSAC, RansacThresh);

    if (!geometry.empty())
    {
        std::vector<cv::Point2f> currCorners;
        cv::transform(prevCorners, currCorners, geometry);

        output = boundingRect2f(currCorners); //Update rectangle

        if (output.width < 0 || output.height < 0) //wrong estimation
            return false;

        cv::Rect shrinkedObj = shrinkRect(output);

        //Handle the box partially outside the frame
        cv::Rect2f intersection = findIntersection(shrinkedObj);
        if (intersection.area()/shrinkedObj.area() < 0.5)
            return false;
        else
            shrinkedObj = intersection;

        prevKeypoints = detectGridFASTpoints(currFrameGray, shrinkedObj); //Update keypoints

        currFrameGray.copyTo(prevGrayFrame);
        prevCorners = currCorners;

        return true;
    }
}


//----------------------------------------------------------------------------
//----------------------------------------------------------------------------
cv::Rect2f FastKltTracker::findIntersection(const cv::Rect2f& input)
{
    cv::Rect2f frame = cv::Rect2f(0.f,0.f,prevGrayFrame.cols, prevGrayFrame.rows);
    return (input & frame);
}


//----------------------------------------------------------------------------
//----------------------------------------------------------------------------
cv::Rect FastKltTracker::shrinkRect(const cv::Rect& input)
{
    cv::Rect output = input;

    cv::Size deltaSize( input.width * shrinkRatio, input.height * shrinkRatio ); // 0.1f = 10/100

    cv::Point offset( deltaSize.width/2, deltaSize.height/2);
    output -= deltaSize;
    output += offset;

    return output;
}


//----------------------------------------------------------------------------
//----------------------------------------------------------------------------
cv::Rect2f FastKltTracker::boundingRect2f(const std::vector<cv::Point2f>& points)
{
    float x[4] = {points[0].x, points[1].x, points[2].x, points[3].x};
    float y[4] = {points[0].y, points[1].y, points[2].y, points[3].y};

    float min_x = *std::min_element(x, x+4);
    float max_x = *std::max_element(x, x+4);

    float min_y = *std::min_element(y, y+4);
    float max_y = *std::max_element(y, y+4);

    float width = max_x - min_x;
    float height = max_y - min_y;

    return cv::Rect2f(min_x, min_y, width, height);
}


//----------------------------------------------------------------------------
//----------------------------------------------------------------------------
struct ResponseComparator
{
    bool operator() (const cv::KeyPoint& a, const cv::KeyPoint& b)
    {
        return std::abs(a.response) > std::abs(b.response);
    }
};


//----------------------------------------------------------------------------
//----------------------------------------------------------------------------
void FastKltTracker::keepStrongest( int N, std::vector<cv::KeyPoint>& keypoints)
{
    if( (int)keypoints.size() > N )
    {
        std::vector<cv::KeyPoint>::iterator nth = keypoints.begin() + N;
        std::nth_element( keypoints.begin(), nth, keypoints.end(), ResponseComparator() );
        keypoints.erase( nth, keypoints.end() );
    }
}


//----------------------------------------------------------------------------
//----------------------------------------------------------------------------
std::vector<cv::Point2f> FastKltTracker::detectGridFASTpoints(cv::Mat image, cv::Rect object)
{
    std::vector<cv::Point2f> keypoints;

    if (object.x < 0 || object.y < 0 || object.width < 0 || object.height < 0 || object.x+object.width > image.cols || object.y+object.height > image.rows)
        return keypoints;

    image = image(object);

    int gridRows = grid.width;
    int gridCols = grid.height;

    keypoints.reserve(maxTotalKeypoints);

    int maxPerCell = maxTotalKeypoints / (gridRows * gridCols);

    for( int i = 0; i < gridRows; ++i )
    {
        cv::Range row_range((i*image.rows)/gridRows, ((i+1)*image.rows)/gridRows);
        for( int j = 0; j < gridCols; ++j )
        {
            cv::Range col_range((j*image.cols)/gridCols, ((j+1)*image.cols)/gridCols);
            cv::Mat sub_image = image(row_range, col_range);

            std::vector<cv::KeyPoint> sub_keypoints;
            cv::FAST(sub_image, sub_keypoints, FastThresh);
            keepStrongest( maxPerCell, sub_keypoints );

            for (size_t it = 0; it<sub_keypoints.size(); it++)
            {
                float x = float(sub_keypoints[it].pt.x + col_range.start+object.x);
                float y = float(sub_keypoints[it].pt.y + row_range.start+object.y);
                keypoints.push_back(cv::Point2f(x,y));
            }
        }
    }

    return keypoints;
}


//----------------------------------------------------------------------------
//----------------------------------------------------------------------------
float FastKltTracker::track(const cv::Mat &img_1, const cv::Mat &img_2, std::vector<cv::Point2f>& points1, std::vector<cv::Point2f>& points2)
{
    std::vector<uchar> status;
    cv::calcOpticalFlowPyrLK(img_1, img_2, points1, points2, status, cv::noArray(), KltWinSize, 3);

    //getting rid of points for which the KLT tracking failed or those who have gone outside the frame
    unsigned int indexCorrection = 0;
    cv::Point2f pt;
    for( size_t i=0; i<status.size(); i++)
    {
        pt = points2[i- indexCorrection];
        if ((status[i] == 0)||(pt.x<0)||(pt.y<0))
        {
            points1.erase (points1.begin() + i - indexCorrection);
            points2.erase (points2.begin() + i - indexCorrection);
            indexCorrection++;
        }

    }

    return 1.f*points1.size()/status.size();
}
