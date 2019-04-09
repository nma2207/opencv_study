#include "lesson5.h"

#include <opencv2/opencv.hpp>

using namespace cv;

namespace  {

Mat getReImage(Mat complex)
{
    Mat result[] = {Mat::zeros(complex.size(), CV_32FC1),
                    Mat::zeros(complex.size(),  CV_32FC1)};
    split(complex, result);

    Mat reImage;

    result[0].convertTo(reImage, CV_8U);


    return  reImage;
}

int opencvFunction(const char* filename)
{

    Mat I = imread(filename, IMREAD_GRAYSCALE);
    if( I.empty())
        return -1;

    Mat padded;                            //expand input image to optimal size
    int m = getOptimalDFTSize( I.rows );
    int n = getOptimalDFTSize( I.cols ); // on the border add zero values
    copyMakeBorder(I, padded, 0, m - I.rows, 0, n - I.cols, BORDER_CONSTANT, Scalar::all(0));

    Mat planes[] = {Mat_<float>(padded), Mat::zeros(padded.size(), CV_32F)};
    Mat complexI;
    merge(planes, 2, complexI);         // Add to the expanded another plane with zeros

    dft(complexI, complexI);            // this way the result may fit in the source matrix

    // compute the magnitude and switch to logarithmic scale
    // => log(1 + sqrt(Re(DFT(I))^2 + Im(DFT(I))^2))
    split(complexI, planes);                   // planes[0] = Re(DFT(I), planes[1] = Im(DFT(I))
    magnitude(planes[0], planes[1], planes[0]);// planes[0] = magnitude
    Mat magI = planes[0];

    magI += Scalar::all(1);                    // switch to logarithmic scale
    log(magI, magI);

    // crop the spectrum, if it has an odd number of rows or columns
    magI = magI(Rect(0, 0, magI.cols & -2, magI.rows & -2));

    // rearrange the quadrants of Fourier image  so that the origin is at the image center
    int cx = magI.cols/2;
    int cy = magI.rows/2;

    Mat q0(magI, Rect(0, 0, cx, cy));   // Top-Left - Create a ROI per quadrant
    Mat q1(magI, Rect(cx, 0, cx, cy));  // Top-Right
    Mat q2(magI, Rect(0, cy, cx, cy));  // Bottom-Left
    Mat q3(magI, Rect(cx, cy, cx, cy)); // Bottom-Right

    Mat tmp;                           // swap quadrants (Top-Left with Bottom-Right)
    q0.copyTo(tmp);
    q3.copyTo(q0);
    tmp.copyTo(q3);

    q1.copyTo(tmp);                    // swap quadrant (Top-Right with Bottom-Left)
    q2.copyTo(q1);
    tmp.copyTo(q2);

    normalize(magI, magI, 0, 1, NORM_MINMAX); // Transform the matrix with float values into a
                                            // viewable image form (float between values 0 and 1).

    imshow("Input Image"       , I   );    // Show the result
    imshow("spectrum magnitude", magI);
    waitKey();

    return 0;
}

} // namespace

void lesson5::main()
{
    //opencvFunction("lena1.jpg");
    //return;
    Mat image = imread("text3.png", IMREAD_GRAYSCALE);


    Mat planes[] = {Mat_<float>(image), Mat::zeros(image.size(), CV_32FC1)};

    Mat complexI;
    merge(planes, 2, complexI);
    Mat result;
    dft(complexI, result);
    split(result, planes);                   // planes[0] = Re(DFT(I), planes[1] = Im(DFT(I))

    //magnitude(planes[0], planes[1], planes[0]);// planes[0] = magnitude
    result = planes[0];

    result += Scalar::all(1);                    // switch to logarithmic scale
    log(result, result);


    int mx = result.size().width / 2;
    int my = result.size().height / 2;
    //imshow("Fasd", getReImage(result));

    //log(result, result);

    Mat a = Mat(result, Rect(0,0, mx, my));
    Mat b = Mat(result, Rect(mx, my, mx, my));

    Mat tmp;
    a.copyTo(tmp);
    b.copyTo(a);
    tmp.copyTo(b);

    Mat c = Mat(result, Rect(0, my, mx, my));
    Mat d = Mat(result, Rect(mx, 0, mx, my));

    c.copyTo(tmp);
    d.copyTo(c);
    tmp.copyTo(d);

    normalize(result, result, 0, 1, NORM_MINMAX);
    imshow("Fourie", result);


    waitKey(0);
}

/*
 * Fourie transform
 *
 * x = 0..M-1, y = 0..N-1
 * f(x,y) - image
 * F(u, v) = sum by x sum by y f(x,y)*exp{-2PI*i*(xu+yu)/MN}
 *
 * f(x,y) = 1/MN sum by u sum by v F(u,v) *exp{-//- with minus}
 *
 * |F(u,v)| = sqrt(Re(F(u,v))^2 + Im(F(u,v))^2
 *
 */
