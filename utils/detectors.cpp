/*
 * detectors.cpp
 *
 *  Created on: Jul 24, 2017
 *      Author: austin
 */

#include "detectors.hpp"

using namespace std;
using namespace cv;


void vl_covariant_detector(Mat image, vector<KeyPoint>& keypoints, VlCovDetMethod type, bool affine_transformation) {
	VlCovDet * covdet = vl_covdet_new(type);

	vl_covdet_put_image(covdet, (float *)image.data, image.rows, image.cols);
	vl_covdet_detect(covdet);
	if(affine_transformation) {
		vl_covdet_extract_affine_shape(covdet);
	}
	vl_covdet_extract_orientations(covdet);

	vl_size numFeatures = vl_covdet_get_num_features(covdet);
	VlCovDetFeature *feature = (VlCovDetFeature *)vl_covdet_get_features(covdet);

	keypoints.clear();

	cout << "numFeatures: " << numFeatures << "\n";

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
		double angle_rad = atan2(- (c - a + sqrt((c-a)*(c-a) + b*b)), -b);
		double angle_deg = angle_rad / PI * 180 + 180;

		keypoints.push_back(KeyPoint(feature[i].frame.x, feature[i].frame.y, size, angle_deg));
	}
}


void DogFeatureDetector(Mat image, vector<KeyPoint>& keypoints) {
	vl_covariant_detector(image, keypoints, VL_COVDET_METHOD_DOG, false);
}


void HesLapFeatureDetector(Mat image, vector<KeyPoint>& keypoints) {
	vl_covariant_detector(image, keypoints, VL_COVDET_METHOD_HESSIAN_LAPLACE, false);
}


void HarLapFeatureDetector(Mat image, vector<KeyPoint>& keypoints) {
	vl_covariant_detector(image, keypoints, VL_COVDET_METHOD_HARRIS_LAPLACE, false);
}


void HesAffFeatureDetector(Mat image, vector<KeyPoint>& keypoints) {
	vl_covariant_detector(image, keypoints, VL_COVDET_METHOD_HESSIAN_LAPLACE, true);
}


void HarAffFeatureDetector(Mat image, vector<KeyPoint>& keypoints) {
	vl_covariant_detector(image, keypoints, VL_COVDET_METHOD_HARRIS_LAPLACE, true);
}


