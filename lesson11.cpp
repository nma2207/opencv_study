#include "lesson11.h"

#include <opencv2/opencv.hpp>

#include <iostream>

using namespace cv;

namespace  {

int t1 = 100;
int t2 = 100;
Mat *image;
void contours(int, void*)
{
    Mat canny;

    Mat img;
    image->copyTo(img);
    Canny(img, canny, t1, t2);
    //imshow("count", canny);

    std::vector<std::vector<Point> > contours;
    std::vector<Vec4i> hierarchy;
    findContours(canny, contours, hierarchy, RETR_TREE,CHAIN_APPROX_SIMPLE, Point(0, 0));

    //std::cout << contours.size() << " "<<hierarchy.size() << std::endl;
    Mat drawing = Mat::zeros(canny.size(), CV_8UC1);
    Mat img1;
    img.copyTo(img1);
    for (int i = 0; i < contours.size(); i++ ) {
        Scalar color = Scalar(255, 255, 255);
        drawContours(drawing, contours, i, color, 3, 8, hierarchy, 0, Point());
        drawContours(img1, contours, i, Scalar(255, 0,0 ), 1, 8, hierarchy, 0, Point());

    }
    bitwise_not(drawing, drawing);

    //imshow("Contours", drawing);

    Mat labelImage(img.size(), CV_32S);
    int nLabels = connectedComponents(drawing, labelImage, 4);
    std::cout <<" labels " << nLabels << std::endl;
    std::vector<Vec3b> colors(nLabels);
    colors[0] = Vec3b(0, 0, 0);//background
    for(int label = 1; label < nLabels; ++label){
        colors[label] = Vec3b( (rand()&255), (rand()&255), (rand()&255) );
    }
    Mat dst(img.size(), CV_8UC3);
    for(int r = 0; r < dst.rows; ++r){
        for(int c = 0; c < dst.cols; ++c){
            int label = labelImage.at<int>(r, c);
            Vec3b &pixel = dst.at<Vec3b>(r, c);
            pixel = colors[label];
         }
     }

    Mat images[] = {img1, dst};
    Mat res;
    //hconcat(images, 2, res);
    imshow( "Contours", dst );

}

void conneComp()
{
    int threshval = 100;
    Mat img = imread("sq.png");
    Mat bw = threshval < 128 ? (img < threshval) : (img > threshval);
    Mat labelImage(img.size(), CV_32S);
    int nLabels = connectedComponents(bw, labelImage, 8);
    std::vector<Vec3b> colors(nLabels);
    colors[0] = Vec3b(0, 0, 0);//background
    for(int label = 1; label < nLabels; ++label){
        colors[label] = Vec3b( (rand()&255), (rand()&255), (rand()&255) );
    }
    Mat dst(img.size(), CV_8UC3);
    for(int r = 0; r < dst.rows; ++r){
        for(int c = 0; c < dst.cols; ++c){
            int label = labelImage.at<int>(r, c);
            Vec3b &pixel = dst.at<Vec3b>(r, c);
            pixel = colors[label];
         }
     }
    imshow( "Connected Components", dst );
    waitKey(0);
}
} // namespace

void lesson11::main()
{
    //conneComp();
    //return;
    Mat image1 = imread("sq.png", IMREAD_GRAYSCALE);
    image = &image1;
    namedWindow("Contours");
    createTrackbar("   t1", "Contours", &t1, 2000, contours);
    createTrackbar("   t2", "Contours", &t2, 2000, contours);
    contours(0,0);
    waitKey(0);
}
