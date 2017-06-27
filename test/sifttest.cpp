/*
 * displayimage.cpp
 *
 *  Created on: Jun 23, 2017
 *      Author: austin
 *
 *  Goal of program is to confirm that I can successfully interface with OpenCV libraries.
 */

#include <iostream>
#include <fstream>
#include "opencv2/opencv.hpp"
#include "opencv2/xfeatures2d.hpp"

using namespace cv;

int main( int argc, char** argv ) {
    Mat image, descriptors;
    std::vector<KeyPoint> keypoints;

    if(argc == 2) {
		image = imread(argv[1], CV_LOAD_IMAGE_COLOR);
		if(!image.data) {
			std::cout << "No image data \n";
			return -1;
		}
	}
    else {
		std::cout << "Incorrect number of arguments.\n";
		return -1;
	}

    Ptr<Feature2D> f2d = xfeatures2d::SIFT::create(); // use default parameters
    f2d->detect(image, keypoints); // detect keypoints
    f2d->compute(image, keypoints, descriptors); // extract descriptors

    std::ofstream myfile;
	myfile.open("siftoutput.csv");
	myfile << cv::format(descriptors, cv::Formatter::FMT_CSV) << std::endl;
	myfile.close();

//    BFMatcher matcher;
//    std::vector< DMatch > matches;
//    matcher.match( descriptors_1, descriptors_2, matches );

    return 0;
}
