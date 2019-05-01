#include "lesson8.h"

#include <opencv2/opencv.hpp>

#include <iostream>

using namespace cv;

namespace {

struct GFTTInfo
{
    Mat* image;
    int* qualityLevel;
    int* minDistance;
};

void gftt(int, void* data)
{
    GFTTInfo* useful = static_cast<GFTTInfo*>(data);
    Mat image = *useful->image;
    double quality = static_cast<double>(*useful->qualityLevel) / 20;
    double minDistance = static_cast<double>(*useful->minDistance) / 10;
    Mat dstGFTT;
    goodFeaturesToTrack(image, dstGFTT, 10, quality, minDistance);

    Mat result;
    image.convertTo(result, CV_8UC3);
//    cvtColor(image, result, COLOR_GRAY2BGR);
//    for (int i = 0; i< dstGFTT.size[0]; i++) {
//        int x = static_cast<int>(dstGFTT.at<float>(i, 0));
//        int y = static_cast<int>(dstGFTT.at<float>(i, 1));
//        circle(result, {x, y}, 5, Scalar{0, 255, 0}, FILLED);
//    }
    auto x = ORB::create();
    std::vector<KeyPoint> points;
    x->detect(image, points);

    Mat out;
    x->compute(image, points, out);
    std::cout << points.size() << std::endl;


    imshow("GFTT", out);

    waitKey(0);
}

} // namespace

void lesson8::main()
{
    Mat image = imread("lena1.jpg", IMREAD_GRAYSCALE);
    // void cornerHarris(InputArray src, OutputArray dst, int blockSize, int ksize, double k, int borderType=BORDER_DEFAULT )¶
    //Mat dstHarris;
    //cornerHarris(image, dstHarris, 5,5,5);
    //imshow("Harris", dstHarris);
   //waitKey(0);
    int quality = 20;
    int distance = 1;
    GFTTInfo info{&image, &quality, &distance};

    namedWindow("GFTT");
    createTrackbar("Quality", "GFTT", &quality, 20, gftt, &info);
    createTrackbar("Distance", "GFTT", &distance, 10, gftt, &info);

    gftt(0, &info);

    waitKey(0);



}
