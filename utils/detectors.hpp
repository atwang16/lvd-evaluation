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

extern "C" {
#include <vl/covdet.h>
}

using namespace std;
using namespace cv;

#define PI 3.14159

void vl_covariant_detector(Mat image, vector<KeyPoint>& keypoints, VlCovDetMethod type, bool affine_transformation);

void DogFeatureDetector(Mat image, vector<KeyPoint>& keypoints);

void HesLapFeatureDetector(Mat image, vector<KeyPoint>& keypoints);

void HarLapFeatureDetector(Mat image, vector<KeyPoint>& keypoints);

void HesAffFeatureDetector(Mat image, vector<KeyPoint>& keypoints);

void HarAffFeatureDetector(Mat image, vector<KeyPoint>& keypoints);

#endif /* UTILS_DETECTORS_HPP_ */
