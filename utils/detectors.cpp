/*
 * detectors.cpp
 *
 *  Created on: Jul 24, 2017
 *      Author: Austin Wang
 *      Project: A Comparative Study of Local Visual Descriptors
 *
 *  Contains implementations of functions to generate keypoints from images.
 */

#include "detectors.hpp"

using namespace boost::filesystem;
using namespace std;
using namespace cv;


Mat get_patch(Mat image, int patch_size, Point2f pt, Mat affine) {
	Mat image_fl, aff_image = Mat::zeros(image.rows, image.cols, CV_32F), patch;
	if(image.type() != CV_32F) {
		image.convertTo(image_fl, CV_32F);
	}
	else {
		image_fl = image;
	}
	warpAffine(image_fl, aff_image, affine, aff_image.size());
	getRectSubPix(aff_image, cv::Size(patch_size, patch_size), pt, patch);
	return patch;
}


Mat get_patch(Mat image, int patch_size, int center_x, int center_y, Mat affine) {
	return get_patch(image, patch_size, Point2f(center_x, center_y), affine);
}


void vl_covariant_detector(Mat image, KeyPointCollection& kp_col, VlCovDetMethod type, bool affine_transformation, float peak_threshold) {
	VlCovDet * covdet = vl_covdet_new(type);
	Mat image_fl;
	image.convertTo(image_fl, CV_32F);

	vl_covdet_set_peak_threshold(covdet, peak_threshold);
	vl_covdet_put_image(covdet, (float *)image_fl.data, image.rows, image.cols);
	vl_covdet_detect(covdet);

	if(affine_transformation) {
		vl_covdet_extract_affine_shape(covdet);
	}
	vl_covdet_extract_orientations(covdet);

	vl_size numFeatures = vl_covdet_get_num_features(covdet);
	VlCovDetFeature *feature = (VlCovDetFeature *)vl_covdet_get_features(covdet);

	kp_col.keypoints.clear();

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
			kp_col.keypoints.push_back(KeyPoint(feature[i].frame.x, feature[i].frame.y, size, angle_deg));
			Mat trans = Mat::zeros(2, 3, CV_32F);
			trans.at<float>(0, 0) = feature[i].frame.a11;
			trans.at<float>(0, 1) = feature[i].frame.a12;
			trans.at<float>(0, 2) = 0;
			trans.at<float>(1, 0) = feature[i].frame.a21;
			trans.at<float>(1, 1) = feature[i].frame.a22;
			trans.at<float>(1, 2) = 0;
			kp_col.affine.push_back(trans);
		}
	}
}

void difference_of_gaussians(Mat image, KeyPointCollection& kp_col, string parameter_file) {
	map<string, double> params = {{"peak_threshold", 0.0}};

	load_parameters(parameter_file, params);

	vl_covariant_detector(image, kp_col, VL_COVDET_METHOD_DOG, false, params["peak_threshold"]);
}


void hessian_laplace(Mat image, KeyPointCollection& kp_col, string parameter_file) {
	map<string, double> params = {{"peak_threshold", 0.0}};

	load_parameters(parameter_file, params);

	vl_covariant_detector(image, kp_col, VL_COVDET_METHOD_HESSIAN_LAPLACE, false, params["peak_threshold"]);
}


void harris_laplace(Mat image, KeyPointCollection& kp_col, string parameter_file) {
	map<string, double> params = {{"peak_threshold", 0.0}};

	load_parameters(parameter_file, params);

	vl_covariant_detector(image, kp_col, VL_COVDET_METHOD_HARRIS_LAPLACE, false, params["peak_threshold"]);
}


void hessian_affine(Mat image, KeyPointCollection& kp_col, string parameter_file) {
	map<string, double> params = {{"peak_threshold", 0.0}};

	load_parameters(parameter_file, params);

	vl_covariant_detector(image, kp_col, VL_COVDET_METHOD_HESSIAN_LAPLACE, true, params["peak_threshold"]);
}


void harris_affine(Mat image, KeyPointCollection& kp_col, string parameter_file) {
	map<string, double> params = {{"peak_threshold", 0.0}};

	load_parameters(parameter_file, params);

	vl_covariant_detector(image, kp_col, VL_COVDET_METHOD_HARRIS_LAPLACE, true, params["peak_threshold"]);
}


void shi_tomasi(Mat image, KeyPointCollection& kp_col, string parameter_file) {
	map<string, double> params = {{"max_corners", 1000},
								  {"quality_level", 0.01},
								  {"min_distance", 1.0},
								  {"block_size", 3},
								  {"use_harris_detector", 0},
								  {"k", 0.04}};

	load_parameters(parameter_file, params);

	Ptr< cv::GFTTDetector > shi_tomasi = GFTTDetector::create(params["max_corners"], params["quality_level"],
			params["min_distance"], params["block_size"], dtob(params["use_harris_detector"]), params["k"]);
	shi_tomasi->detect(image, kp_col.keypoints);
}


void censure(Mat image, KeyPointCollection& kp_col, string parameter_file) {
	map<string, double> params = {{"max_size", 45},
								  {"response_threshold", 30},
								  {"line_threshold_projected", 10},
								  {"line_threshold_binarized", 8},
								  {"suppress_nonmax_size", 5}};

	load_parameters(parameter_file, params);

	Ptr< xfeatures2d::StarDetector > censure = xfeatures2d::StarDetector::create(params["max_size"], params["response_threshold"],
			params["line_threshold_projected"], params["line_threshold_binarized"], params["suppress_nonmax_size"]);
	censure->detect(image, kp_col.keypoints);
}


void mser(Mat image, KeyPointCollection& kp_col, string parameter_file) {
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
	mser->detect(image, kp_col.keypoints);
}


void fast(Mat image, KeyPointCollection& kp_col, string parameter_file) {
	map<string, double> params = {{"fast_threshold", 10},
								  {"nonmax_suppression", 1},
								  {"type", FastFeatureDetector::TYPE_9_16}};

	load_parameters(parameter_file, params);

	Ptr< cv::FastFeatureDetector > fast = FastFeatureDetector::create(params["fast_threshold"], params["nonmax_suppression"],
			params["type"]);
	fast->detect(image, kp_col.keypoints);
}


void agast(Mat image, KeyPointCollection& kp_col, string parameter_file) {
	map<string, double> params = {{"agast_threshold", 10},
								  {"nonmax_suppression", 1},
								  {"type", AgastFeatureDetector::OAST_9_16}};

	load_parameters(parameter_file, params);

	Ptr<Feature2D> agast = AgastFeatureDetector::create(params["agast_threshold"], params["nonmax_suppression"],
			params["type"]);
	agast->detect(image, kp_col.keypoints);
}

void brisk(Mat image, KeyPointCollection& kp_col, string parameter_file) {
	map<string, double> params = {{"agast_threshold", 30},
								  {"n_octaves", 3},
								  {"pattern_scale", 1.0}};

	load_parameters(parameter_file, params);

	cv::Ptr<cv::Feature2D> brisk =
			cv::BRISK::create(params["agast_threshold"], params["n_octaves"], params["pattern_scale"]);
	brisk->detect(image, kp_col.keypoints);
}

void kaze(Mat image, KeyPointCollection& kp_col, string parameter_file) {
	map<string, double> params = {{"extended", 0},
								  {"upright", 0},
								  {"threshold", 0.001f},
								  {"n_octaves", 4},
								  {"n_octave_layers", 4},
								  {"diffusivity", cv::KAZE::DIFF_PM_G2}};

	load_parameters(parameter_file, params);

	cv::Ptr<cv::Feature2D> kaze =
			cv::KAZE::create(dtob(params["extended"]), dtob(params["upright"]), params["threshold"],
					params["n_octaves"], params["n_octave_layers"], params["diffusivity"]);
	kaze->detect(image, kp_col.keypoints);
}

void orb(cv::Mat image, KeyPointCollection& kp_col, std::string parameter_file) {
	map<string, double> params = {{"n_features", 10000},
								  {"scale_factor", 1.2},
								  {"n_levels", 8},
								  {"edge_threshold", 31},
								  {"first_level", 0},
								  {"wta_k", 2},
								  {"score_type", cv::ORB::HARRIS_SCORE},
								  {"patch_size", 31},
								  {"fast_threshold", 20}};

	load_parameters(parameter_file, params);

	cv::Ptr<cv::Feature2D> orb = cv::ORB::create(params["n_features"], params["scale_factor"],
			params["n_levels"],params["edge_thresh"], params["first_level"], params["wta_k"],
			params["score_type"], params["patch_size"], params["fast_thresh"]);
	orb->detect(image, kp_col.keypoints);
}

void sift(Mat image, KeyPointCollection& kp_col, string parameter_file) {
	map<string, double> params = {{"n_features", 0},
								  {"n_octave_layers", 3},
								  {"contrast_threshold", 0.04},
								  {"edge_threshold", 10},
								  {"sigma", 1.6}};

	load_parameters(parameter_file, params);

	cv::Ptr<cv::Feature2D> sift = cv::xfeatures2d::SIFT::create(params["n_features"], params["n_octave_layers"],
			params["contrast_threshold"], params["edge_threshold"], params["sigma"]);
	sift->detect(image, kp_col.keypoints);
}

void surf(cv::Mat image, KeyPointCollection& kp_col, std::string parameter_file) {
	map<string, double> params = {{"hessian_threshold", 100},
								  {"n_octaves", 4},
								  {"n_octave_layers", 3},
								  {"extended", 0}};
	bool upright = false;

	load_parameters(parameter_file, params);
	params["upright"] = 0;

	cv::Ptr<cv::Feature2D> surf = cv::xfeatures2d::SURF::create(params["hessian_threshold"], params["n_octaves"],
			params["n_octave_layers"], dtob(params["extended"]), upright);
	surf->detect(image, kp_col.keypoints);
}

void usurf(cv::Mat image, KeyPointCollection& kp_col, std::string parameter_file) {
	map<string, double> params = {{"hessian_threshold", 100},
								  {"n_octaves", 4},
								  {"n_octave_layers", 3},
								  {"extended", 0}};
	bool upright = true;

	load_parameters(parameter_file, params);

	cv::Ptr<cv::Feature2D> usurf = cv::xfeatures2d::SURF::create(params["hessian_threshold"], params["n_octaves"],
			params["n_octave_layers"], dtob(params["extended"]), upright);
	usurf->detect(image, kp_col.keypoints);
}

