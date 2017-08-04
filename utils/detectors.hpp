/*
 * detectors.hpp
 *
 *  Created on: Jul 25, 2017
 *      Author: austin
 */

#ifndef UTILS_DETECTORS_HPP_
#define UTILS_DETECTORS_HPP_

#include "utils.hpp"
#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <map>
#include <cmath>

extern "C" {
#include <vl/covdet.h>
}

struct KeyPointCollection {
	std::vector<cv::KeyPoint> keypoints;
	std::vector<cv::Mat> affine;
};

typedef void (*Detector)(cv::Mat, KeyPointCollection&, std::string);

#define PI 3.14159

#define DETECTOR_MAP {"shi_tomasi", shi_tomasi},       \
					 {"censure", censure},             \
					 {"mser", mser},                   \
					 {"fast", fast},                   \
					 {"agast", agast},                 \
					 {"dog", difference_of_gaussians}, \
					 {"hesaff", hessian_affine},       \
					 {"haraff", harris_affine},        \
					 {"heslap", hessian_laplace},      \
					 {"harlap", harris_laplace},       \
					 {"sift", sift}

#define detector_args cv::Mat image, KeyPointCollection& keypoints, std::string parameter_file

cv::Mat get_patch(cv::Mat image, int patch_size, cv::Point2f pt, cv::Mat affine);

cv::Mat get_patch(cv::Mat image, int patch_size, int center_x, int center_y, cv::Mat affine);

void difference_of_gaussians(detector_args);

void hessian_laplace(detector_args);

void harris_laplace(detector_args);

void hessian_affine(detector_args);

void harris_affine(detector_args);

void shi_tomasi(detector_args);

void censure(detector_args);

void mser(detector_args);

void fast(detector_args);

void agast(detector_args);

void sift(detector_args);

#endif /* UTILS_DETECTORS_HPP_ */
