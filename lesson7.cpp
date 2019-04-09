#include "lesson7.h"

#include <opencv2/opencv.hpp>

#include <iostream>

using namespace cv;

namespace  {

struct WienerInfo
{
    Mat *image;
    int *nsr;
    int *blurSize;
    int *dBlurSize;
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

Mat getBlur(Size size, int width)
{
    Mat blur = Mat::zeros(size, CV_32F);

    int cx = blur.size().width / 2;
    int cy = blur.size().height / 2;
    circle(blur, Point{cx, cy}, width, 255, -1, 8);
    Scalar summa = sum(blur);
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

    Mat doubleImage;
    image.convertTo(doubleImage, CV_32F);


    Mat blur = getBlur(image.size(), m);
    swap(blur);

    Mat F = getDft(image);
    Mat H = getDft(blur);
//    Mat reF = getReImage(F, 0, CV_32FC1);
//    Mat reH = getReImage(H, 0, CV_32FC1);
//    Mat imF = getReImage(F, 1, CV_32FC1);
//    Mat imH = getReImage(H, 1, CV_32FC1);

//    //multiply(reF, reH, reF);
//    //multiply(imF, imH, imF);
//    Mat f[] = {reF, imF};
//    merge(f, 2, F);
//    Mat h[] = {reH, imH};
//    merge(h, 2, H);



    //multiply(F, H, F);
    mulSpectrums(F, H, F, 0, false);

    blur = getBlur(image.size(), dm);
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

    imshow("deconvolution", getReImage(res, 0, CV_8U));
}

} // namespace

void lesson7::main()
{

    Mat image = imread("lena1.jpg", IMREAD_GRAYSCALE);

    int k = 0;
    int blurSize = 1;
    int dBlurSize = 1;
    WienerInfo info{&image, &k, &blurSize, &dBlurSize};

    namedWindow("deconvolution");
    createTrackbar("Nsr", "deconvolution", &k, 20000, deconvolution, &info);
    createTrackbar("real radius", "deconvolution", &blurSize, 10, deconvolution, &info);
    createTrackbar("dec. radius", "deconvolution", &dBlurSize, 10, deconvolution, &info);

    deconvolution(0, &info);

    waitKey(0);
}
