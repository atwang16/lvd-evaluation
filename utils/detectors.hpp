/*
 * detectors.hpp
 *
 *  Created on: Jul 25, 2017
 *      Author: austin
 */

#ifndef UTILS_DETECTORS_HPP_
#define UTILS_DETECTORS_HPP_

#include <opencv2/opencv.hpp>
#include <cmath>
#include <boost/filesystem.hpp>
#include <boost/algorithm/string.hpp>
#include "opencv2/xfeatures2d.hpp"

extern "C" {
#include <vl/covdet.h>
}

using namespace std;
using namespace cv;

#define PI 3.14159


enum Detector {
	SHI_TOMASI = 0,
	CENSURE,
	MSER,
	FAST,
	DIFFERENCE_OF_GAUSSIANS,
	HESSIAN_LAPLACE,
	HARRIS_LAPLACE,
	HESSIAN_AFFINE,
	HARRIS_AFFINE,
	END_OF_DETECTORS,
	SIFT,
};


void vl_covariant_detector(Mat image, vector<KeyPoint>& keypoints, VlCovDetMethod type, bool affine_transformation);

void DogFeatureDetector(Mat image, vector<KeyPoint>& keypoints);

void HesLapFeatureDetector(Mat image, vector<KeyPoint>& keypoints);

void HarLapFeatureDetector(Mat image, vector<KeyPoint>& keypoints);

void HesAffFeatureDetector(Mat image, vector<KeyPoint>& keypoints);

void HarAffFeatureDetector(Mat image, vector<KeyPoint>& keypoints);

#endif /* UTILS_DETECTORS_HPP_ */
