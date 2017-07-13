/*
 * utils.hpp
 *
 *  Created on: Jul 12, 2017
 *      Author: austin
 */

#ifndef EVALUATION_UTILS_HPP_
#define EVALUATION_UTILS_HPP_

#include <opencv2/opencv.hpp>
#include <string>
#include <iostream>
#include <sstream>
#include <fstream>

cv::Mat parse_file(std::string fname, char delimiter, int type);

int get_dist_metric(std::string metric);

bool is_overlapping(cv::KeyPoint kp_1, cv::KeyPoint kp_2, cv::Mat hom, float threshold);


#endif /* EVALUATION_UTILS_HPP_ */
