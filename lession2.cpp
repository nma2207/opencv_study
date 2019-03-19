#include "lession2.h"

#include <opencv2/opencv.hpp>

using namespace cv;

namespace  {

Mat getHistogram(Mat& image, int channel)
{
    int hbins = 256, sbins = 256;
    int histSize[] = {hbins, sbins};
    // hue varies from 0 to 179, see cvtColor
    float hranges[] = { 0, 256 };
    // saturation varies from 0 (black-gray-white) to
    // 255 (pure spectrum color)
    float sranges[] = { 0, 256 };
    const float* ranges[] = { hranges, sranges };
    MatND hist;
    int channels[] = {channel};
    // we compute the histogram from the 0-th and 1-st channels

    calcHist( &image, 1, {channels}, Mat(), // do not use mask
             hist, 1, histSize, ranges,
             true, // the histogram is uniform
             false );

    double maxVal=0;
    minMaxLoc(hist, 0, &maxVal, 0, 0);

    int scalar = 1;
    Mat histImg = Mat::zeros(sbins*scalar, hbins*scalar, CV_8UC3);


    Scalar color{255 * static_cast<double>(channel == 0),
                255 * static_cast<double>(channel == 1),
                255 * static_cast<double>(channel == 2)};
    for( int h = 0; h < 1; h++ )
        for( int s = 0; s < sbins; s++ )
        {
            float binVal = hist.at<float>(h, s);
            int intensity = cvRound(binVal*255/maxVal);
            rectangle( histImg, Point(s * scalar, hbins * scalar),
                        Point( (s+1)*scalar, (hbins-intensity)*scalar),
                        color,
                        FILLED );
        }

    return histImg;
}

} // namespace

void lession2::main()
{
    Mat image = imread("lena1.jpg");
    Mat histograms{};

    std::cout << "dim "<<histograms.dims<<std::endl;
    for (int channel = 0; channel < 3; channel++) {
        Mat hist = getHistogram(image, channel);

        if (histograms.dims != 0) {
            hconcat(histograms, hist, histograms);
        }
        else {
            hist.copyTo(histograms);
        }
    }

    imshow("Histogram", histograms);

    waitKey(0);
}
