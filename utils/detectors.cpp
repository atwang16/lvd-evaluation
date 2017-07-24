/*
 * detectors.cpp
 *
 *  Created on: Jul 24, 2017
 *      Author: austin
 */

#include <opencv2/opencv.hpp>
#include "opencv2/xfeatures2d.hpp"
#include <string>
#include <iostream>
#include <sstream>
#include <fstream>
#include <cmath>

extern "C" {
#include <vl/covdet.h>
}

using namespace std;
using namespace cv;

void DogFeatureDetector(Mat image, vector<KeyPoint>& keypoints) {
	vl_covariant_detector(image, keypoints, VL_COVDET_METHOD_DOG);
}

void vl_covariant_detector(Mat image, vector<KeyPoint>& keypoints, VlCovDetMethod type) {
	VlCovDet *covdet = vl_covdet_new(type);

	vl_covdet_put_image(covdet, (float *)image.data, image.rows, image.cols);
	vl_covdet_detect(covdet);
	vl_covdet_extract_orientations(covdet);

	vl_size numFeatures = vl_covdet_get_num_features(covdet);
	VlCovDetFeature const *feature = vl_covdet_get_features(covdet);

	if(keypoints.size() > 0) {
		keypoints.clear();
	}

	for(int i = 0; i < numFeatures; i++) {
		Mat aff = Mat::zeros(3, 3, CV_32F);
		aff.at<float>(0, 0) = feature[i].frame.a11 * feature[i].frame.a11 + feature[i].frame.a21 * feature[i].frame.a21;
		aff.at<float>(0, 1) = feature[i].frame.a11 * feature[i].frame.a12 + feature[i].frame.a21 * feature[i].frame.a22;
		aff.at<float>(0, 2) = feature[i].frame.x   * feature[i].frame.a11 + feature[i].frame.y   * feature[i].frame.a21;
		aff.at<float>(1, 0) = feature[i].frame.a11 * feature[i].frame.a12 + feature[i].frame.a21 * feature[i].frame.a22;
		aff.at<float>(1, 1) = feature[i].frame.a12 * feature[i].frame.a12 + feature[i].frame.a22 * feature[i].frame.a22;
		aff.at<float>(1, 2) = feature[i].frame.x   * feature[i].frame.a12 + feature[i].frame.y   * feature[i].frame.a22;
		aff.at<float>(2, 0) = feature[i].frame.x   * feature[i].frame.a11 + feature[i].frame.y   * feature[i].frame.a21;
		aff.at<float>(2, 1) = feature[i].frame.x   * feature[i].frame.a12 + feature[i].frame.y   * feature[i].frame.a22;
		aff.at<float>(2, 2) = feature[i].frame.x   * feature[i].frame.x   + feature[i].frame.y   * feature[i].frame.y   - 1;

		double size = pow(determinant(aff), 0.5);
		float a = aff.at<float>(0, 0), b = aff.at<float>(0, 1) / 2.0, c = aff.at<float>(1, 1);
		double angle = atan2(c - a + sqrt((c-a)*(c-a) + b*b), b);

		keypoints.push_back(KeyPoint(feature[i].frame.x, feature[i].frame.y, size, angle));
	}
}

