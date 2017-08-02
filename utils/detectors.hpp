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
#include <boost/assign/list_of.hpp>
#include "opencv2/xfeatures2d.hpp"
#include <map>
#include "utils.hpp"

extern "C" {
#include <vl/covdet.h>
}

using namespace std;
using namespace cv;

#define PI 3.14159


void difference_of_gaussians(Mat image, vector<KeyPoint>& keypoints, string parameter_file);

void hessian_laplace(Mat image, vector<KeyPoint>& keypoints, string parameter_file);

void harris_laplace(Mat image, vector<KeyPoint>& keypoints, string parameter_file);

void hessian_affine(Mat image, vector<KeyPoint>& keypoints, string parameter_file);

void harris_affine(Mat image, vector<KeyPoint>& keypoints, string parameter_file);

void shi_tomasi(Mat image, vector<KeyPoint>& keypoints, string parameter_file);

void censure(Mat image, vector<KeyPoint>& keypoints, string parameter_file);

void mser(Mat image, vector<KeyPoint>& keypoints, string parameter_file);

void fast(Mat image, vector<KeyPoint>& keypoints, string parameter_file);

void agast(Mat image, vector<KeyPoint>& keypoints, string parameter_file);

Mat get_patch(Mat image, int patch_size, Point2f pt);

Mat get_patch(Mat image, int patch_size, int center_x, int center_y);

#endif /* UTILS_DETECTORS_HPP_ */
