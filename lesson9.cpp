#include "lesson9.h"

#include <opencv2/opencv.hpp>

#include <iostream>

using namespace cv;

namespace {

} // namespace

void lesson9::main()
{
    auto orb = BRISK::create(30, 3, 10);
    //auto orb = AKAZE::create();
     // auto orb = ORB::create(15);

    Mat image = imread("lena1.jpg", IMREAD_GRAYSCALE);
    Mat desc1;
    std::vector<KeyPoint> points1;
    //getDescriptors(image, desc1, points1);
    orb->detect(image, points1);
    orb->compute(image, points1, desc1);

    Mat image2 = imread("lena1.jpg");
    Mat desc2;
    std::vector<KeyPoint> points2;

    orb->detect(image, points2);
    orb->compute(image, points2, desc2);

    auto bfMatcher = BFMatcher::create(NORM_HAMMING);

    std::vector<std::vector<DMatch>> matches;
    //bfMatcher->match(desc1, desc2, matches);
    bfMatcher->knnMatch(desc1, desc2, matches, 5);
    Mat out;
    std::cout << matches.size() << std::endl;
    drawMatches(image, points1, image2, points2, matches, out);
    imshow("ASD", out);
    waitKey(0);

}

// create rotateMatrix2d Mp = p' , M - is transformation matrix
