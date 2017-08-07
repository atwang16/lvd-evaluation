/*
 * descriptors.hpp
 *
 *  Created on: Aug 3, 2017
 *      Author: austin
 */

#ifndef UTILS_DESCRIPTORS_HPP_
#define UTILS_DESCRIPTORS_HPP_

#include "detectors.hpp"
#include "utils.hpp"
#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <map>
#include <chrono>
#include <algorithm>

extern "C" {
#include <vl/liop.h>
}

typedef void (*Descriptor)(cv::Mat, std::vector<cv::KeyPoint>&, std::vector<cv::Mat>&, cv::Mat&, std::string);

#define DESCRIPTOR_ARGS cv::Mat image, std::vector<cv::KeyPoint>& keypoints, std::vector<cv::Mat>& affine, \
	cv::Mat& descriptors, std::string parameter_file

#define DESCRIPTOR_MAP {"akaze", akaze}, \
					   {"brief", brief}, \
					   {"brisk", brisk}, \
					   {"cslbp", cslbp}, \
					   {"freak", freak}, \
					   {"kaze", kaze},   \
					   {"latch", latch}, \
					   {"liop", liop},   \
					   {"lucid", lucid}, \
					   {"orb", orb},     \
					   {"sift", sift},   \
					   {"surf", surf},   \
					   {"usurf", usurf}


void akaze(DESCRIPTOR_ARGS);

void brief(DESCRIPTOR_ARGS);

void brisk(DESCRIPTOR_ARGS);

void cslbp(DESCRIPTOR_ARGS);

void freak(DESCRIPTOR_ARGS);

void kaze(DESCRIPTOR_ARGS);

void latch(DESCRIPTOR_ARGS);

void liop(DESCRIPTOR_ARGS);

void lucid(DESCRIPTOR_ARGS);

void orb(DESCRIPTOR_ARGS);

void sift(DESCRIPTOR_ARGS);

void surf(DESCRIPTOR_ARGS);

void usurf(DESCRIPTOR_ARGS);

#endif /* UTILS_DESCRIPTORS_HPP_ */
