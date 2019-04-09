#include "lesson6.h"

#include <opencv2/opencv.hpp>

#include <iostream>

using namespace cv;

namespace  {

struct DenoiseInfo
{
    Mat& image;
    int& sigmaS;
    int& sigmaN;
};

Mat getReImage(Mat complex)
{
    Mat result[] = {Mat::zeros(complex.size(), CV_32FC1),
                    Mat::zeros(complex.size(),  CV_32FC1)};
    split(complex, result);

    Mat reImage;

    result[0].convertTo(reImage, CV_8U);
    return  reImage;
}

void denoise(int k, void *data)
{

    Mat image = *static_cast<Mat*>(data);

    double nsr = static_cast<double>(k) / 100;

    Mat planes[] = {Mat_<float>(image), Mat::zeros(image.size(), CV_32FC1)};

    Mat complexI;
    merge(planes, 2, complexI);

    dft(complexI, complexI);

    double mu = sum(image)[0];

    Mat tmp[] = {Mat::zeros(complexI.size(), CV_32FC1),
                        Mat::zeros(complexI.size(),  CV_32FC1)};

    complexI = complexI *  (1 / (1+nsr));
    split(complexI, tmp);

    tmp[0].at<double>(0) += mu / ( 1 + 1/nsr);
    merge(tmp, 2, complexI);

    dft(complexI, complexI, DFT_SCALE | DFT_INVERSE);

    std::cout << nsr <<" "<< mu<<" "<< mu / ( 1 + 1/nsr)<<std::endl;
    Mat re = getReImage(complexI);
    imshow("denoise", getReImage(complexI));
}

} // namespace

void lesson6::main()
{
    Mat image = imread("lena1.jpg", IMREAD_GRAYSCALE);

    int k = 0;

    namedWindow("denoise");
    createTrackbar("Nsr", "denoise", &k, 2000, denoise, &image);

    denoise(k, &image);

    waitKey(0);


}

/**
 * f(x) = g(x) + e(x) f <-> F; g <-> G, e <-> E
 * F(t) = G(t) + E(t)
 *
 *
 */
