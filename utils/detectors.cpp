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

void difference_of_gaussians(Mat image, vector<KeyPoint>& keypoints, string parameter_file) {
	vl_covariant_detector(image, keypoints, VL_COVDET_METHOD_DOG, false);
}


void hessian_laplace(Mat image, vector<KeyPoint>& keypoints, string parameter_file) {
	vl_covariant_detector(image, keypoints, VL_COVDET_METHOD_HESSIAN_LAPLACE, false);
}


void harris_laplace(Mat image, vector<KeyPoint>& keypoints, string parameter_file) {
	vl_covariant_detector(image, keypoints, VL_COVDET_METHOD_HARRIS_LAPLACE, false);
}


void hessian_affine(Mat image, vector<KeyPoint>& keypoints, string parameter_file) {
	vl_covariant_detector(image, keypoints, VL_COVDET_METHOD_HESSIAN_LAPLACE, true);
}


void harris_affine(Mat image, vector<KeyPoint>& keypoints, string parameter_file) {
	vl_covariant_detector(image, keypoints, VL_COVDET_METHOD_HARRIS_LAPLACE, true);
}


void shi_tomasi(Mat image, vector<KeyPoint>& keypoints, string parameter_file) {
	map<string, double> params = {{"max_corners", 1000},
								  {"quality_level", 0.01},
								  {"min_distance", 1.0},
								  {"block_size", 3},
								  {"use_harris_detector", 0},
								  {"k", 0.04}};

	load_parameters(parameter_file, params);

	Ptr< cv::GFTTDetector > shi_tomasi = GFTTDetector::create(params["max_corners"], params["quality_level"],
			params["min_distance"], params["block_size"], dtob(params["use_harris_detector"]), params["k"]);
	shi_tomasi->detect(image, keypoints);
}


void censure(Mat image, vector<KeyPoint>& keypoints, string parameter_file) {
	map<string, double> params = {{"max_size", 45},
								  {"response_threshold", 30},
								  {"line_threshold_projected", 10},
								  {"line_threshold_binarized", 8},
								  {"suppress_nonmax_size", 5}};

	load_parameters(parameter_file, params);

	Ptr< xfeatures2d::StarDetector > censure = xfeatures2d::StarDetector::create(params["max_size"], params["response_threshold"],
			params["line_threshold_projected"], params["line_threshold_binarized"], params["suppress_nonmax_size"]);
	censure->detect(image, keypoints);
}


void mser(Mat image, vector<KeyPoint>& keypoints, string parameter_file) {
	map<string, double> params = {{"delta", 5},
								  {"min_area", 60},
								  {"max_area", 14400},
								  {"max_variation", 0.25},
								  {"min_diversity", 0.2},
								  {"max_evolution", 200},
								  {"area_threshold", 1.01},
								  {"min_margin", 0.003},
								  {"edge_blur_size", 5}};

	load_parameters(parameter_file, params);

	Ptr< cv::MSER > mser = cv::MSER::create(params["delta"], params["min_area"], params["max_area"], params["max_variation"],
			params["min_diversity"], params["max_evoluation"], params["area_threshold"],
			params["min_margin"], params["edge_blur_size"]);
	mser->detect(image, keypoints);
}


void fast(Mat image, vector<KeyPoint>& keypoints, string parameter_file) {
	map<string, double> params = {{"fast_threshold", 10},
								  {"nonmax_suppression", 1},
								  {"type", FastFeatureDetector::TYPE_9_16}};

	load_parameters(parameter_file, params);

	Ptr< cv::FastFeatureDetector > fast = FastFeatureDetector::create(params["fast_threshold"], params["nonmax_suppression"],
			params["type"]);
	fast->detect(image, keypoints);
}


void agast(Mat image, vector<KeyPoint>& keypoints, string parameter_file) {
	map<string, double> params = {{"agast_threshold", 10},
								  {"nonmax_suppression", 1},
								  {"type", AgastFeatureDetector::OAST_9_16}};

	load_parameters(parameter_file, params);

	Ptr<Feature2D> agast = AgastFeatureDetector::create(params["agast_threshold"], params["nonmax_suppression"],
			params["type"]);
}


Mat get_patch(Mat image, int patch_size, Point2f pt) {
	Mat image_fl, patch;
	if(image.type() != CV_32F) {
		image.convertTo(image_fl, CV_32F);
	}
	else {
		image_fl = image;
	}
	getRectSubPix(image_fl, cv::Size(patch_size, patch_size), pt, patch);
	return patch;
}


Mat get_patch(Mat image, int patch_size, int center_x, int center_y) {
	return get_patch(image, patch_size, Point2f(center_x, center_y));
}
