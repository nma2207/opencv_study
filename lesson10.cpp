#include "lesson10.h"

#include <opencv2/opencv.hpp>

#include <iostream>
#include <cmath>

using namespace cv;

namespace {

struct CannyInfo
{
    Mat* image;
    int* t1;
    int* t2;
    int* angle;
};

void canny(int, void* data)
{
    CannyInfo* info = static_cast<CannyInfo*>(data);
    Mat image = *info->image;
    double t1 = *info->t1;
    double t2 = *info->t2;
    double angle = *info->angle;
    Mat transform = getRotationMatrix2D({256,256}, angle, 1);
    Mat transformed;
    warpAffine(image, transformed, transform, image.size());
    Mat res1;
    Canny(image, res1, t1, t2, 3, false);
    Mat res2;
    Canny(transformed, res2, t1, t2, 3, false);
    Mat res;
    Mat images[] = {res1, res2};
    hconcat(images, 2, res);

    std::vector<Vec2f> hRes;
    HoughLines(res1, hRes, 1, 1, 10);

//    std::cout << hRes.size() << std::endl;
//    for (int i = 0; i < hRes.size() - 1; i++) {
//        double d = hRes[i][0];
//        double theta = hRes[i][1];
//        double r1 = cos(theta);
//        double r2 = -sin(theta);
//        int x1 = d*cos(theta);
//        int y1 =d*sin(theta);
//        int x2 = 10 * r1 + x1;
//        int y2 = 10 * r2 + x2;
//        line(image, {x1, y1}, {x2, y2}, Scalar{0,255,0});
//    }
// r = cos theta , -sin theta
    // (x,y) = t * r + (x0, y0)
    // (d, theta)
    // (d*cos theta, d * sin theta) = (x0, y0)

    imshow("canny", res);
}

} // namespace

void lesson10::main()
{
    Mat image = imread("lena1.jpg"/*, IMREAD_GRAYSCALE*/);
    int t1 = 166;
    int t2 = 114;
    int angle = 0;
    CannyInfo info{&image, &t1, &t2, &angle};

    namedWindow("canny");
    createTrackbar("   t1", "canny", &t1, 2000, canny, &info);
    createTrackbar("   t2", "canny", &t2, 2000, canny, &info);
    createTrackbar("angle", "canny", &angle, 360, canny, &info);

    canny(0, &info);
    waitKey(0);
}
