#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/video.hpp"
#include "boost/filesystem.hpp"
// someday...#include <filesystem>
//#include "opencv2/videoio.hpp"
#include <iostream>

const char *image_window = "Source Image";
const char *result_window = "Result window";

void maskOn(cv::Mat matchArea, const cv::Mat& cboxmask, cv::Mat& card)
{
    // checkbox mask
    cv::Mat checksonly = cv::Mat::zeros(matchArea.size(), matchArea.type());
    matchArea.copyTo(checksonly, cboxmask);

    cv::imshow(image_window, checksonly);
}

cv::Mat rotateImg(cv::Mat src, int deg) {
    // get rotation matrix for rotating the image around its center
    cv::Point center(src.cols/2.0, src.rows/2.0);
    cv::Mat rot = cv::getRotationMatrix2D(center, deg, 1.0);
    // determine bounding rectangle
    cv::Rect bbox = cv::RotatedRect(center, src.size(), deg).boundingRect();
    // adjuct transformation matrix
    rot.at<double>(0,2) += bbox.width/2.0 - center.x;
    rot.at<double>(1,2) += bbox.height/2.0 - center.y;

    cv::Mat dst;
    cv::warpAffine(src, dst, rot, bbox.size());
    return dst;
}

cv::Mat matchCard(int method, const cv::Mat& cboxes, cv::Mat& card)
{
    int maxAngle = 5;

    double maxVals [maxAngle * 2];
    cv::Point matchLocs [maxAngle * 2];
    for (int deg = -maxAngle; deg <= maxAngle; deg++) {
        // create result matrix
        int result_cols =  card.cols - cboxes.cols + 1;
        int result_rows = card.rows - cboxes.rows + 1;
        cv::Mat result(result_rows, result_cols, CV_32FC1);

        // rotate obj to match
        cv::Mat cboxrot = rotateImg(cboxes,deg);

        // Do the Matching and Normalize
        cv::matchTemplate(card, cboxrot, result, method);
        // cv::normalize(result, result, 0, 1, cv::NORM_MINMAX, -1, cv::Mat());

        // find best match
        double minVal; double maxVal;
        cv::Point minLoc;
        cv::Point maxLoc;

        cv::minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc, cv::Mat());

        // store rotation's match
        maxVals[deg + maxAngle] = maxVal;
        matchLocs[deg + maxAngle] = maxLoc;
    }

    // find best rotation match
    double curMax = 0;
    cv::Point matchLoc;
    int angle = 0;
    std::cout << "maxVals ";
    for (int i = 0; i < maxAngle * 2; i++)
        std::cout << maxVals[i] << " ";
    std::cout << std::endl;
    for (int i = 0; i <= maxAngle * 2; i++) {
        double val = maxVals[i];
        if (val > curMax) {
            curMax = val;
            matchLoc = matchLocs[i];
            angle = i - maxAngle;
        }
    }
    std::cout << "angle " << angle << std::endl;

    cv::Mat cboxrot = rotateImg(cboxes, angle);
    // get center
    cv::Point center(matchLoc.x + cboxrot.cols/2.0, matchLoc.y + cboxrot.rows/2.0);
    cv::Mat M = cv::getRotationMatrix2D(center, angle, 1.0);
    cv::Mat rotated, cropped;
    cv::warpAffine(card, rotated, M, card.size(), cv::INTER_CUBIC);
    cv::getRectSubPix(rotated, cboxes.size(), center, cropped);
    cv::imshow(image_window, cropped);
    return cropped;
}

int main(int argc, char* argv[])
{
    if (argc < 2)
    {
        std::cout << "Usage: cbox_finder path /path/to/checkboxes.png /path/to/cboxmask.png\n";
        return -1;
    }

    std::cout << "Built with OpenCV " << CV_VERSION << std::endl;

    boost::filesystem::path p (argv[1]);
    if (!is_directory(p))
    {
        std::cout << "First arg must be directory." << std::endl;
        return -1;
    }

    boost::filesystem::path checkbox (argv[2]);
    if (!is_regular_file(checkbox))
    {
        std::cout << "Second arg must be file." << std::endl;
        return -1;
    }

    boost::filesystem::path checkboxmask (argv[3]);
    if (!is_regular_file(checkboxmask))
    {
        std::cout << "Third arg must be file." << std::endl;
        return -1;
    }

    cv::Mat cboxes = cv::imread(checkbox.string(), CV_LOAD_IMAGE_GRAYSCALE);
    if (cboxes.empty()) // Check for invalid input
    {
        std::cout << "Could not open or find checkboxes.png" << std::endl;
        return -1;
    }

    cv::Mat cboxmask = cv::imread(checkboxmask.string(), CV_LOAD_IMAGE_GRAYSCALE);
    if (cboxmask.empty()) // Check for invalid input
    {
        std::cout << "Could not open or find cboxmask.png" << std::endl;
        return -1;
    }
    cboxmask = cboxmask > 128;

    auto sz = cboxes.size();
    std::cout << sz.width << "x" << sz.height << std::endl;

    cv::namedWindow(image_window, CV_WINDOW_NORMAL);
    // cv::namedWindow(result_window, CV_WINDOW_NORMAL);

    boost::filesystem::directory_iterator end_itr;
    for (boost::filesystem::directory_iterator itr(p);
            itr != end_itr; itr++)
    {
        auto& path = itr->path();
        if (path.extension() == ".png" &&
                path.stem().string().find("LafayetteI_") == 0)
        {
            std::cout << itr->path().string() << std::endl;
            cv::Mat card = cv::imread(
                    itr->path().string(),
                    CV_LOAD_IMAGE_GRAYSCALE);
            if (card.empty()) // Check for invalid input
            {
                std::cout << "Could not open or find the image" << std::endl;
                return -1;
            }

            cv::Mat matchArea = matchCard(cv::TM_CCOEFF, cboxes, card);
            cv::waitKey(1000);
            maskOn(matchArea, cboxmask, card);
            cv::waitKey(1000);
        }
    }

    return 0;
}
