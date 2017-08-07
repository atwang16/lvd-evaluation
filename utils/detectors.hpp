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
					 {"sift", sift},                   \
					 {"brisk", brisk},                 \
					 {"kaze", kaze},                   \
					 {"orb", orb},                     \
					 {"surf", surf},                   \
					 {"usurf", usurf}

#define DETECTOR_ARGS cv::Mat image, KeyPointCollection& keypoints, std::string parameter_file

cv::Mat get_patch(cv::Mat image, int patch_size, cv::Point2f pt, cv::Mat affine);

cv::Mat get_patch(cv::Mat image, int patch_size, int center_x, int center_y, cv::Mat affine);

void difference_of_gaussians(DETECTOR_ARGS);

void hessian_laplace(DETECTOR_ARGS);

void harris_laplace(DETECTOR_ARGS);

void hessian_affine(DETECTOR_ARGS);

void harris_affine(DETECTOR_ARGS);

void shi_tomasi(DETECTOR_ARGS);

void censure(DETECTOR_ARGS);

void mser(DETECTOR_ARGS);

void fast(DETECTOR_ARGS);

void agast(DETECTOR_ARGS);

void sift(DETECTOR_ARGS);

void brisk(DETECTOR_ARGS);

void kaze(DETECTOR_ARGS);

void orb(DETECTOR_ARGS);

void surf(DETECTOR_ARGS);

void usurf(DETECTOR_ARGS);

#endif /* UTILS_DETECTORS_HPP_ */
