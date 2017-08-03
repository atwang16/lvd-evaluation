/*
 * detectors.hpp
 *
 *  Created on: Jul 25, 2017
 *      Author: austin
 */

#ifndef UTILS_DETECTORS_HPP_
#define UTILS_DETECTORS_HPP_

#ifndef UTILS_UTILS_HPP_
#include "utils.hpp"
#endif

#include <cmath>
#include <boost/assign/list_of.hpp>
#include "opencv2/xfeatures2d.hpp"
#include <map>

extern "C" {
#include <vl/covdet.h>
}

struct KeyPointCollection {
	std::vector<cv::KeyPoint> keypoints;
	std::vector<cv::Mat> affine;
};

#define PI 3.14159

#define DETECTORS {"SHI_TOMASI", shi_tomasi},       \
                  {"CENSURE", censure},             \
				  {"MSER", mser},                   \
				  {"FAST", fast},                   \
				  {"AGAST", agast},                 \
				  {"DOG", difference_of_gaussians}, \
				  {"HESAFF", hessian_affine},       \
				  {"HARAFF", harris_affine},        \
				  {"HESLAP", hessian_laplace},      \
				  {"HARLAP", harris_laplace},       \
				  {"SIFT", sift}

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
