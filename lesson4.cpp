#include "lesson4.h"

#include <opencv2/opencv.hpp>


using namespace cv;

namespace  {

struct SharepingInfo
{
    int *k;
    int *alpha;
    Mat *image;
};

void sharepering(int, void* data) {
    auto info = static_cast<SharepingInfo*>(data);

    assert(info->k);
    assert(info->alpha);
    assert(info->image);

    Mat image = *info->image;
    Mat blurA;
    int size = 2 * (*info->k) + 1;
    double a = static_cast<double>(*info->alpha) / 100;

    GaussianBlur(image, blurA, Size{size, size}, 5);

    Mat shar;

    addWeighted(image, (1+a), blurA, -a, 0, shar);

    imshow("Shareped image", shar);
}

void openShareperingWindow()
{
    Mat image = imread("lena1.jpg");
    int k = 1;
    int alpha = 1;
    SharepingInfo info{&k, &alpha, &image};

    namedWindow("Shareped image");
    createTrackbar("Blur size", "Shareped image", &k, 10, sharepering, &info);
    createTrackbar("Alpha", "Shareped image", &alpha, 1000, sharepering, &info);
    sharepering(0, &info);
    waitKey(0);
}

struct BilateralInfo
{
    int* distance;
    int* sigmaColor;
    int* sigmaSpace;
    Mat* image;
};

void bilateralFilter(int, void* bilateralInfo)
{
    auto info = static_cast<BilateralInfo*>(bilateralInfo);
    Mat image = *info->image;
    Mat filtered;

    int d = *info->distance;
    double sigmaColor = static_cast<double>(*info->sigmaColor) / 10;
    double sigmaSpace = static_cast<double>(*info->sigmaSpace) / 10;
    bilateralFilter(image, filtered, d, sigmaColor, sigmaSpace);
    imshow("Bilateral filter", filtered);
}

void openBilateralFilterWindow()
{
    int distance = 1;
    int sigmaColor = 10;
    int sigmaSpace = 10;
    Mat image = imread("lena.jpeg");
    BilateralInfo info{&distance, &sigmaColor, &sigmaSpace, &image};

    namedWindow("Bilateral filter");
    createTrackbar("D", "Bilateral filter", &distance, 10, bilateralFilter, &info);
    createTrackbar("Sigma color", "Bilateral filter", &sigmaColor, 10000, bilateralFilter,
                   &info);
    createTrackbar("Sigma space", "Bilateral filter", &sigmaSpace, 150, bilateralFilter,
                   &info);
    bilateralFilter(0, &info);
    waitKey(0);
}

struct MorphologyExInfo
{
    int* morphOperator; //0..7
    int* shape;//0..2
    int* size;// = 3;
    Mat* image;
};

void morphologyEx(int, void* data)
{
    auto info = static_cast<MorphologyExInfo*>(data);
    Mat image = *info->image;

//    MORPH_ERODE    = 0, //!< see #erode
//    MORPH_DILATE   = 1, //!< see #dilate
//    MORPH_OPEN     = 2, //!< an opening operation
//                        //!< \f[\texttt{dst} = \mathrm{open} ( \texttt{src} , \texttt{element} )= \mathrm{dilate} ( \mathrm{erode} ( \texttt{src} , \texttt{element} ))\f]
//    MORPH_CLOSE    = 3, //!< a closing operation
//                        //!< \f[\texttt{dst} = \mathrm{close} ( \texttt{src} , \texttt{element} )= \mathrm{erode} ( \mathrm{dilate} ( \texttt{src} , \texttt{element} ))\f]
//    MORPH_GRADIENT = 4, //!< a morphological gradient
//                        //!< \f[\texttt{dst} = \mathrm{morph\_grad} ( \texttt{src} , \texttt{element} )= \mathrm{dilate} ( \texttt{src} , \texttt{element} )- \mathrm{erode} ( \texttt{src} , \texttt{element} )\f]
//    MORPH_TOPHAT   = 5, //!< "top hat"
//                        //!< \f[\texttt{dst} = \mathrm{tophat} ( \texttt{src} , \texttt{element} )= \texttt{src} - \mathrm{open} ( \texttt{src} , \texttt{element} )\f]
//    MORPH_BLACKHAT = 6,

//    MORPH_RECT    = 0, //!< a rectangular structuring element:  \f[E_{ij}=1\f]
//    MORPH_CROSS   = 1, //!< a cross-shaped structuring element:
//                       //!< \f[E_{ij} =  \fork{1}{if i=\texttt{anchor.y} or j=\texttt{anchor.x}}{0}{otherwise}\f]
//    MORPH_ELLIPSE = 2

    Mat res;
    int morphOperator = *info->morphOperator;
    int shape = *info->shape;
    int size = *info->size + 1;
    morphologyEx(image, res, morphOperator,
                 getStructuringElement(shape, Size{size, size}));
    imshow("Morphology Ex", res);
}

void openMorphologyExWindow()
{
    int morphOperator = 0;
    int shape = 0;
    int size = 3;
    Mat image = imread("lena1.jpg");
    MorphologyExInfo info{&morphOperator, &shape, &size, &image};

    namedWindow("Morphology Ex");
    createTrackbar("Operator", "Morphology Ex", &morphOperator, 6, morphologyEx, &info);
    createTrackbar("Shape", "Morphology Ex", &shape, 2, morphologyEx, &info);
    createTrackbar("Size", "Morphology Ex", &size, 15, morphologyEx, &info);
    morphologyEx(0, &info);
    waitKey(0);
}

} // namespace

void lesson4::main()
{
    //openShareperingWindow();
    //openBilateralFilterWindow();
    openMorphologyExWindow();

}
