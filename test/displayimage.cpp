/*
 * displayimage.cpp
 *
 *  Created on: Jun 23, 2017
 *      Author: austin
 *
 *  Goal of program is to confirm that I can successfully interface with OpenCV libraries.
 */

#include <iostream>
#include "opencv2/opencv.hpp"

#define WIDTH 640
#define HEIGHT 480

using namespace cv;

int main( int argc, char** argv ) {
    Mat image, dst;

    if(argc == 2) {
        image = imread(argv[1], CV_LOAD_IMAGE_COLOR);
        if(!image.data) {
            std::cout << "No image data \n";
            return -1;
        }
        else {
        	resize(image, dst, Size(), 0.5, 0.5, INTER_AREA);
        }
    }
    else {
        std::cout << "Incorrect number of arguments.\n";
        return -1;
    }

    namedWindow("Display Image", WINDOW_NORMAL); // Create window for image
    resizeWindow("Display Image", WIDTH, HEIGHT);
    imshow("Display Image", dst); // show image in window
    waitKey(0); // wait until key is pressed
    return 0;
}
