/*
 * descriptors.hpp
 *
 *  Created on: Aug 3, 2017
 *      Author: austin
 */

#ifndef UTILS_DESCRIPTORS_HPP_
#define UTILS_DESCRIPTORS_HPP_

#ifndef UTILS_UTILS_HPP_
#include "utils.hpp"
#endif

#include "opencv2/xfeatures2d.hpp"
#include <chrono>
#include <algorithm>

extern "C" {
#include <vl/liop.h>
}

#define DESCRIPTORS {"AKAZE", akaze}, \
                    {"BRIEF", brief}, \
				    {"BRISK", brisk}, \
				    {"CSLBP", cslbp}, \
				    {"FREAK", freak}, \
				    {"KAZE", kaze},   \
				    {"LATCH", latch}, \
				    {"LIOP", liop},   \
				    {"LUCID", lucid}, \
				    {"ORB", orb},     \
				    {"SIFT", sift},   \
					{"SURF", surf},   \
					{"USURF", usurf}

#define descriptor_args cv::Mat image, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors, std::string parameter_file

void akaze(descriptor_args);

void brief(descriptor_args);

void brisk(descriptor_args);

void cslbp(descriptor_args);

void freak(descriptor_args);

void kaze(descriptor_args);

void latch(descriptor_args);

void liop(descriptor_args);

void lucid(descriptor_args);

void orb(descriptor_args);

void sift(descriptor_args);

void surf(descriptor_args);

void usurf(descriptor_args);

#endif /* UTILS_DESCRIPTORS_HPP_ */
