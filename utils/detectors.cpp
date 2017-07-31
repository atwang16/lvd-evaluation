/*
 * detectors.cpp
 *
 *  Created on: Jul 24, 2017
 *      Author: austin
 */

#include "detectors.hpp"

using namespace boost::filesystem;
using namespace std;
using namespace cv;


void vl_covariant_detector(Mat image, vector<KeyPoint>& keypoints, VlCovDetMethod type, bool affine_transformation) {
	VlCovDet * covdet = vl_covdet_new(type);
	Mat image_fl;
	image.convertTo(image_fl, CV_32F);

	vl_covdet_put_image(covdet, (float *)image_fl.data, image.rows, image.cols);
	vl_covdet_detect(covdet);

	if(affine_transformation) {
		vl_covdet_extract_affine_shape(covdet);
	}
	vl_covdet_extract_orientations(covdet);

	vl_size numFeatures = vl_covdet_get_num_features(covdet);
	VlCovDetFeature *feature = (VlCovDetFeature *)vl_covdet_get_features(covdet);

	keypoints.clear();

	for(int i = 0; i < numFeatures; i++) {
//		if(i < 100)
//			cout << feature[i].edgeScore << ", " << feature[i].peakScore << "\n";

		Mat aff = Mat::zeros(3, 3, CV_32F);
		aff.at<float>(0, 0) = feature[i].frame.a11 * feature[i].frame.a11 + feature[i].frame.a21 * feature[i].frame.a21 + 0;
		aff.at<float>(0, 1) = feature[i].frame.a11 * feature[i].frame.a12 + feature[i].frame.a21 * feature[i].frame.a22 + 0;
		aff.at<float>(0, 2) = feature[i].frame.x   * feature[i].frame.a11 + feature[i].frame.y   * feature[i].frame.a21 + 0;
		aff.at<float>(1, 0) = feature[i].frame.a11 * feature[i].frame.a12 + feature[i].frame.a21 * feature[i].frame.a22 + 0;
		aff.at<float>(1, 1) = feature[i].frame.a12 * feature[i].frame.a12 + feature[i].frame.a22 * feature[i].frame.a22 + 0;
		aff.at<float>(1, 2) = feature[i].frame.x   * feature[i].frame.a12 + feature[i].frame.y   * feature[i].frame.a22 + 0;
		aff.at<float>(2, 0) = feature[i].frame.x   * feature[i].frame.a11 + feature[i].frame.y   * feature[i].frame.a21 + 0;
		aff.at<float>(2, 1) = feature[i].frame.x   * feature[i].frame.a12 + feature[i].frame.y   * feature[i].frame.a22 + 0;
		aff.at<float>(2, 2) = feature[i].frame.x   * feature[i].frame.x   + feature[i].frame.y   * feature[i].frame.y   - 1;

		double size = pow(abs(determinant(aff)), 0.5);
		float a = aff.at<float>(0, 0), b = aff.at<float>(0, 1) / 2.0, c = aff.at<float>(1, 1);
		double angle_rad = atan2(- (c - a + sqrt((c-a)*(c-a) + b*b)), -b);
		double angle_deg = angle_rad / PI * 180 + 180;

		if(size > 50) {
			keypoints.push_back(KeyPoint(feature[i].frame.x, feature[i].frame.y, size, angle_deg));
		}
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


void ShiTomasiFeatureDetector(Mat image, vector<KeyPoint>& keypoints, string parameter_file) {

	vector<string> params = {"max_corners", "quality_level", "min_distance", "block_size", "use_harris_detector", "k"};
	vector<double> values = {1000, 0.01, 1.0, 3, 0, 0.04};

	if(params.size() != values.size()) {
		cout << "Error: mismatch in number of parameters and associated values in detector. Aborting...\n";
		return;
	}

	// Load parameters from file
	std::ifstream parameter_filestream(parameter_file);
	string line, line_split, var, value;
	while(getline(parameter_filestream, line)) {
		boost::split(line_split, line, boost::is_any_of("="));
		var = line_split[0];
		std::transform(var.begin(), var.end(), var.begin(), ::tolower);
		value = line_split[1];

		for(int i = 0; i < params.size(); i++) {
			if(var == params[i]) {
				values[i] = stod(value);
			}
		}
	}

	Ptr< cv::GFTTDetector > shi_tomasi = GFTTDetector::create(values[0], values[1], values[2], values[3], values[4], values[5]);
	shi_tomasi->detect(image, keypoints);
}

void CensureFeatureDetector(Mat image, vector<KeyPoint>& keypoints, string parameter_file) {
	vector<string> params = {"max_size", "response_threshold", "line_threshold_projected", "line_threshold_binarized",
			"suppress_nonmax_size"};
	vector<double> values = {45, 30, 10, 8, 5};

	if(params.size() != values.size()) {
		cout << "Error: mismatch in number of parameters and associated values in detector. Aborting...\n";
		return;
	}

	// Load parameters from file
	std::ifstream parameter_filestream(parameter_file);
	string line, line_split, var, value;
	while(getline(parameter_filestream, line)) {
		boost::split(line_split, line, boost::is_any_of("="));
		var = line_split[0];
		std::transform(var.begin(), var.end(), var.begin(), ::tolower);
		value = line_split[1];

		for(int i = 0; i < params.size(); i++) {
			if(var == params[i]) {
				values[i] = stod(value);
			}
		}
	}

	Ptr< xfeatures2d::StarDetector > censure = xfeatures2d::StarDetector::create(values[0], values[1], values[2], values[3], values[4]);
	censure->detect(image, keypoints);
}


void MserFeatureDetector(Mat image, vector<KeyPoint>& keypoints, string parameter_file) {
	vector<string> params = {"delta", "min_area", "max_area", "max_variation", "min_diversity", "max_evoluation", "area_threshold",
			"min_margin", "edge_blur_size"};
	vector<double> values = {5, 60, 14400, 0.25, 0.2, 200, 1.01, 0.003, 5};

	if(params.size() != values.size()) {
		cout << "Error: mismatch in number of parameters and associated values in detector. Aborting...\n";
		return;
	}

	// Load parameters from file
	std::ifstream parameter_filestream(parameter_file);
	string line, line_split, var, value;
	while(getline(parameter_filestream, line)) {
		boost::split(line_split, line, boost::is_any_of("="));
		var = line_split[0];
		std::transform(var.begin(), var.end(), var.begin(), ::tolower);
		value = line_split[1];

		for(int i = 0; i < params.size(); i++) {
			if(var == params[i]) {
				values[i] = stod(value);
			}
		}
	}

	Ptr< cv::MSER > mser = cv::MSER::create(values[0], values[1], values[2], values[3], values[4], values[5], values[6], values[7], values[8]);
	mser->detect(image, keypoints);
}

void FastFeatureDetector(Mat image, vector<KeyPoint>& keypoints, string parameter_file) {
	vector<string> params = {"threshold", "nonmax_suppression", "type"};
	vector<double> values = {10, 1, FastFeatureDetector::TYPE_9_16};

	if(params.size() != values.size()) {
		cout << "Error: mismatch in number of parameters and associated values in detector. Aborting...\n";
		return;
	}

	// Load parameters from file
	std::ifstream parameter_filestream(parameter_file);
	string line, line_split, var, value;
	while(getline(parameter_filestream, line)) {
		boost::split(line_split, line, boost::is_any_of("="));
		var = line_split[0];
		std::transform(var.begin(), var.end(), var.begin(), ::tolower);
		value = line_split[1];

		for(int i = 0; i < params.size(); i++) {
			if(var == params[i]) {
				values[i] = stod(value);
			}
		}
	}

	Ptr< cv::FastFeatureDetector > fast = cv::FastFeatureDetector::create(values[0], values[1], values[2]);
	fast->detect(image, keypoints);
}

