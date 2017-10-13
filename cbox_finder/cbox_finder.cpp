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

void maskOn(cv::Rect checkboxArea, const cv::Mat& cboxmask, cv::Mat& card)
{
    // crop checkbox region image
    cv::Mat croppedImage = card(checkboxArea);

    // checkbox mask
    cv::Mat checksonly = cv::Mat::zeros(croppedImage.size(), croppedImage.type());
    croppedImage.copyTo(checksonly, cboxmask);

    cv::imshow(image_window, checksonly);
}

cv::Rect matchCard(int method, const cv::Mat& cboxes, cv::Mat& card)
{
    // create result matrix
    int result_cols =  card.cols - cboxes.cols + 1;
    int result_rows = card.rows - cboxes.rows + 1;
    cv::Mat result(result_rows, result_cols, CV_32FC1);

    /// Do the Matching and Normalize
    cv::matchTemplate(card, cboxes, result, method);
    cv::normalize(result, result, 0, 1, cv::NORM_MINMAX, -1, cv::Mat());

    // find best match
    double minVal; double maxVal;
    cv::Point minLoc;
    cv::Point maxLoc;

    cv::minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc, cv::Mat());

    cv::Point matchLoc = maxLoc;

    // show the rect
    cv::rectangle(card, matchLoc,
            cv::Point(matchLoc.x + cboxes.cols, matchLoc.y + cboxes.rows),
            cv::Scalar::all(0), 2, 8, 0);

    cv::imshow(image_window, card);
    // cv::imshow(result_window, result);

    cv::Rect checkboxArea(matchLoc.x, matchLoc.y, cboxes.cols, cboxes.rows);
    return checkboxArea;
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

            cv::Rect matchLoc = matchCard(cv::TM_CCOEFF, cboxes, card);
            cv::waitKey(1000);
            maskOn(matchLoc, cboxmask, card);
            cv::waitKey(1000);
        }
    }

    return 0;
}
