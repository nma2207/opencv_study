#include "lesson1.h"

#include <opencv2/opencv.hpp>

#include <iostream>

using namespace cv;

namespace  {

uchar gammaCorrection(uchar input, double gamma)
{
    return static_cast<uchar>(std::pow(static_cast<double>(input)/255, gamma) * 255);
}

} // namespace

void lesson1::main()
{
    Mat image = imread("lena.jpeg");
    Mat reduced(1, 256, CV_8U);
    Mat lookUpTable(1, 256, CV_8U);

    double gamma = 0.5;
    for( int i = 0; i < 255; ++i)
        lookUpTable.at<uchar>(i) = gammaCorrection(i, gamma);

    LUT(image, lookUpTable, reduced);

    imshow("Gamma correction", reduced);

    waitKey(0);
}
