#include "finalTask.h"

#include <opencv2/opencv.hpp>

#include <iostream>
#include <unistd.h>

using namespace cv;

namespace {

struct WienerInfo
{
    Mat *image;
    int *nsr;
    int *blurSize;
    int *dBlurSize;
    int *blurType;
    int *dBlurType;
};

void swap(Mat& image)
{
    int mx = image.size().width / 2;
    int my = image.size().height / 2;
    Mat a = Mat(image, Rect(0,0, mx, my));
    Mat b = Mat(image, Rect(mx, my, mx, my));

    Mat tmp;
    a.copyTo(tmp);
    b.copyTo(a);
    tmp.copyTo(b);

    Mat c = Mat(image, Rect(0, my, mx, my));
    Mat d = Mat(image, Rect(mx, 0, mx, my));

    c.copyTo(tmp);
    d.copyTo(c);
    tmp.copyTo(d);
}

enum class BlurType {
    Circle = 0,
    Rectangle,
    Gaussian,
    COUNT
};

Mat getBlur(Size size, int width, BlurType blurType)
{
    Mat blur = Mat::zeros(size, CV_32F);

    int cx = blur.size().width / 2;
    int cy = blur.size().height / 2;
    switch (blurType) {
        case BlurType::Circle:
            circle(blur, Point{cx, cy}, width, 255, -1, 8);
            break;
        case BlurType::Rectangle:
            rectangle(blur, Point{cx - width , cy - width},
                      Point{cx + width, cy + width}, 255, -1, 8);
            break;
        case BlurType::Gaussian:
            Mat t = Mat(blur(Rect(cx-width, cy-width, 2*width+1, 2*width+1)));

            Mat gaussian = getGaussianKernel(2 * width + 1, 5);
            Mat gaussianT;
            transpose(gaussian, gaussianT);
            gaussian = gaussian * gaussianT;

            gaussian.convertTo(gaussian, CV_32F);
            gaussian.copyTo(t);
            break;

    }
    Scalar summa = sum(blur);
    //std::cout << summa[0] << std::endl;
    //std::cout << summa[0] << std::endl;
    blur = blur / summa[0];
    //std::cout << sum(blur)[0] << std::endl;
    return blur;
}

Mat getReImage(Mat complex, int channel, int type)
{
    Mat result[] = {Mat::zeros(complex.size(), CV_32FC1),
                    Mat::zeros(complex.size(),  CV_32FC1)};
    split(complex, result);

    Mat reImage;

    result[channel].convertTo(reImage, type);
    return  reImage;
}

Mat getDft(Mat image)
{
    Mat planes[] = {Mat_<float>(image), Mat::zeros(image.size(), CV_32FC1)};

    Mat complex;
    merge(planes, 2, complex);
    dft(complex, complex);

    return complex;
}

void deconvolution(int , void* data)
{
    WienerInfo* info = static_cast<WienerInfo*>(data);
    Mat image = *info->image;
    int k = *info->nsr + 1;
    double nsr = 1 / static_cast<double>(k);
    int m = *info->blurSize;
    int dm = *info->dBlurSize;
    BlurType blurType = static_cast<BlurType>(*info->blurType);
    BlurType dblurType = static_cast<BlurType>(*info->dBlurType);

    Mat doubleImage;
    image.convertTo(doubleImage, CV_32F);


    Mat blur = getBlur(image.size(), m, blurType);
    //imshow("deconvolution", blur);
    //return;
    swap(blur);

    Mat F = getDft(image);
    Mat H = getDft(blur);

    //multiply(F, H, F);
    mulSpectrums(F, H, F, 0, false);

    Mat blurredImgF;
    dft(F, blurredImgF, DFT_INVERSE | DFT_SCALE);
    Mat blurredImg = getReImage(blurredImgF, 0, CV_8U);

    blur = getBlur(image.size(), dm, dblurType);
    swap(blur);


    H = getDft(blur);

    Mat nom;
    mulSpectrums(F, H, nom, 0, true);
    Mat denom;
    pow(abs(H), 2, denom);
    //mulSpectrums(H, H, denom, 0, true);

    Mat reNom = getReImage(nom, 0, CV_32FC1);
    Mat reDenom = getReImage(denom, 0, CV_32FC1);
    Mat reRes;
    divide(reNom, reDenom + nsr , reRes);

    Mat imNom = getReImage(nom, 1, CV_32FC1);
    Mat imDenom = getReImage(denom, 1, CV_32FC1);
    Mat imRes;
    divide(imNom, reDenom + nsr , imRes);

    Mat res;
    Mat p[] = {reRes, imRes};
    merge(p, 2, res);
    //idft(res, res, DFT_SCALE);
    //normalize(res, res, 0, 255, NORM_MINMAX);
    dft(res, res, DFT_INVERSE | DFT_SCALE);

    //imshow("deconvolution", getReImage(res, 0, CV_8U));
    Mat deblurredImg = getReImage(res, 0, CV_8U);
    Mat images[] = {image, blurredImg, deblurredImg};

    Mat result;
    hconcat(images, 3, result);
    imshow("deconvolution", result);

}

} // namespace

void finalTask::wienerFilter()
{
    Mat image = imread("lena1.jpg", IMREAD_GRAYSCALE);

    int k = 10;
    int blurSize = 1;
    int dBlurSize = 1;
    int blurType = 0;
    int dblurType = 0;
    WienerInfo info{&image, &k, &blurSize, &dBlurSize, &blurType, &dblurType};

    namedWindow("deconvolution");
    createTrackbar("1 / nsr", "deconvolution", &k, 20000, deconvolution, &info);
    createTrackbar("real radius", "deconvolution", &blurSize, 10, deconvolution, &info);
    createTrackbar("dec. radius", "deconvolution", &dBlurSize, 10, deconvolution, &info);
    createTrackbar("blurType ", "deconvolution", &blurType,
                   static_cast<int>(BlurType::COUNT) - 1, deconvolution, &info);
    createTrackbar("dblurType", "deconvolution", &dblurType,
                   static_cast<int>(BlurType::COUNT) - 1, deconvolution, &info);

    deconvolution(0, &info);

    waitKey(0);
    //system("pause");

}

namespace  {

}

void finalTask::trackFeatures()
{
    std::cout << "Final Task: track features";
}
