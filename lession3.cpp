#include "lession3.h"

#include <opencv2/opencv.hpp>

using namespace cv;

namespace {

void calcBlur(int k, void* image)
{
    Mat* originalImage = static_cast<Mat*>(image);

    Mat blurImage;

    k = k + 1;
    blur(*originalImage, blurImage, Size{k, k});
    //GaussianBlur(*originalImage, blurImage, Size{k,k}, 5);
    //Laplacian(A, blurA, 3, k);
    //Sobel(A, blurA, 3, 1, 1, k);

    imshow("Blur image", blurImage);
}

} // namespace

void lession3::main()
{
    Mat image = imread("lena1.jpg");

    namedWindow("Blur image");

    int k = 1;
    createTrackbar("Blur size", "Blur image", &k, 10, calcBlur, &image);

    calcBlur(k, &image);
    waitKey(0);
}
