/*
 * descriptors.cpp
 *
 *  Created on: Aug 4, 2017
 *      Author: austin
 */

#include "detectors.hpp"
#include "descriptors.hpp"

using namespace boost::filesystem;
using namespace std;
using namespace cv;


void akaze(cv::Mat image, KeyPointCollection& kp_col, cv::Mat& descriptors, std::string parameter_file) {
	map<string, double> params = {{"descriptor_type", cv::AKAZE::DESCRIPTOR_MLDB},
								  {"descriptor_size", 0},
								  {"descriptor_channels", 3},
								  {"threshold", 0.001},
								  {"n_octaves", 4},
								  {"n_octave_layers", 4},
								  {"diffusivity", cv::KAZE::DIFF_PM_G2}};

	load_parameters(parameter_file, params);

	cv::Ptr<cv::Feature2D> akaze = cv::AKAZE::create(params["descriptor_type"], params["descriptor_size"],
			params["descriptor_channels"], params["threshold"], params["n_octaves"], params["n_octave_layers"],
			params["diffusivity"]);
	akaze->detectAndCompute(image, cv::noArray(), kp_col.keypoints, descriptors, false); // seems to have a bug that causes it to produce different
																						 // descriptors when not computing both simultaneously
}

void binboost(cv::Mat image, KeyPointCollection& kp_col, cv::Mat& descriptors, std::string parameter_file) {
	map<string, double> params = {{"desc", cv::xfeatures2d::BoostDesc::BINBOOST_256},
								  {"use_scale_orientation", 1},
								  {"scale_factor", 6.25}};

	load_parameters(parameter_file, params);

	cv::Ptr<cv::xfeatures2d::BoostDesc> binboost =
			cv::xfeatures2d::BoostDesc::create(dtob(params["use_scale_orientation"]), params["scale_factor"]);
	binboost->compute(image, kp_col.keypoints, descriptors);
}

void brief(cv::Mat image, KeyPointCollection& kp_col, cv::Mat& descriptors, std::string parameter_file) {
	map<string, double> params = {{"bytes", 32},
								  {"use_orientation", 0}};

	load_parameters(parameter_file, params);

	cv::Ptr<cv::xfeatures2d::BriefDescriptorExtractor> brief =
			cv::xfeatures2d::BriefDescriptorExtractor::create(params["bytes"], dtob(params["use_orientation"]));
	brief->compute(image, kp_col.keypoints, descriptors);
}

void brisk(cv::Mat image, KeyPointCollection& kp_col, cv::Mat& descriptors, std::string parameter_file) {
	map<string, double> params = {{"agast_threshold", 30},
								  {"n_octaves", 3},
								  {"pattern_scale", 1.0}};

	load_parameters(parameter_file, params);

	cv::Ptr<cv::Feature2D> brisk =
			cv::BRISK::create(params["agast_threshold"], params["n_octaves"], params["pattern_scale"]);
	brisk->compute(image, kp_col.keypoints, descriptors);
}

void cslbp(cv::Mat image, KeyPointCollection& kp_col, cv::Mat& descriptors, std::string parameter_file) {
	map<string, double> params = {{"threshold", 0.1},
								  {"patch_size", 20}};

	load_parameters(parameter_file, params);

	descriptors = cv::Mat::zeros(kp_col.keypoints.size(), 16, CV_8U);

	for(int i = 0; i < kp_col.keypoints.size(); i++) {
		cv::Mat patch = get_patch(image, params["patch_size"], kp_col.keypoints[i].pt, kp_col.affine[i]);

		for(int x = 1; x < patch.cols - 1; x++) {
			for(int y = 1; y < patch.rows - 1; y++) {
				int a = ((patch.at<float>(x,y+1)   - patch.at<float>(x,y-1)   > params["threshold"]) * 1);
				int b = ((patch.at<float>(x+1,y+1) - patch.at<float>(x-1,y-1) > params["threshold"]) * 2);
				int c = ((patch.at<float>(x+1,y)   - patch.at<float>(x-1,y)   > params["threshold"]) * 4);
				int d = ((patch.at<float>(x+1,y-1) - patch.at<float>(x-1,y+1) > params["threshold"]) * 8);
				descriptors.at<uchar>(i, a+b+c+d)++;
			}
		}
	}
}

void daisy(cv::Mat image, KeyPointCollection& kp_col, cv::Mat& descriptors, std::string parameter_file) {
	map<string, double> params = {{"radius", 15},
								  {"q_radius", 3},
								  {"q_theta", 8},
								  {"q_hist", 8},
								  {"norm", cv::xfeatures2d::DAISY::NRM_NONE},
								  {"interpolation", 1},
								  {"use_orientation", 0}};

	load_parameters(parameter_file, params);

	cv::Ptr<cv::xfeatures2d::DAISY> daisy =
			cv::xfeatures2d::DAISY::create(params["radius"], params["q_radius"], params["q_theta"], params["q_hist"],
					params["norm"], noArray(), dtob(params["interpolation"]), dtob(params["use_orientation"]));
	daisy->compute(image, kp_col.keypoints, descriptors);
}

void freak(cv::Mat image, KeyPointCollection& kp_col, cv::Mat& descriptors, std::string parameter_file) {
	map<string, double> params = {{"orientation_normalized", 1},
								  {"scale_normalized", 1},
								  {"pattern_scale", 22.0f},
								  {"n_octaves", 4}};

	load_parameters(parameter_file, params);

	cv::Ptr<cv::Feature2D> freak =
			cv::xfeatures2d::FREAK::create(dtob(params["orientation_normalized"]), dtob(params["scale_normalized"]),
					params["pattern_scale"], params["n_octaves"]);
	freak->compute(image, kp_col.keypoints, descriptors);
}

void kaze(cv::Mat image, KeyPointCollection& kp_col, cv::Mat& descriptors, std::string parameter_file) {
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
	kaze->compute(image, kp_col.keypoints, descriptors);
}

void latch(cv::Mat image, KeyPointCollection& kp_col, cv::Mat& descriptors, std::string parameter_file) {
	map<string, double> params = {};

	load_parameters(parameter_file, params);

	cv::Ptr<cv::Feature2D> latch = cv::xfeatures2d::LATCH::create();
	latch->compute(image, kp_col.keypoints, descriptors);
}

void liop(cv::Mat image, KeyPointCollection& kp_col, cv::Mat& descriptors, std::string parameter_file) {
	map<string, double> params = {{"patch_size", 41}};

	load_parameters(parameter_file, params);

	VlLiopDesc *liop = vl_liopdesc_new_basic(params["patch_size"]);
	descriptors = cv::Mat::zeros(kp_col.keypoints.size(), vl_liopdesc_get_dimension(liop), CV_8U);

	for(int i = 0; i < kp_col.keypoints.size(); i++) {
		cv::Mat patch = get_patch(image, params["patch_size"], kp_col.keypoints[i].pt, kp_col.affine[i]);
		vl_liopdesc_process(liop, (float *)descriptors.data + i * descriptors.cols, (float *)patch.data);
	}

	vl_liopdesc_delete(liop);
}

void lucid(cv::Mat image, KeyPointCollection& kp_col, cv::Mat& descriptors, std::string parameter_file) {
	map<string, double> params = {{"lucid_kernel", 1},
								  {"blur_kernel", 2}};

	load_parameters(parameter_file, params);

	cv::Ptr<cv::Feature2D> lucid = cv::xfeatures2d::LUCID::create(params["lucid_kernel"], params["blur_kernel"]);
	lucid->compute(image, kp_col.keypoints, descriptors);
}

void orb(cv::Mat image, KeyPointCollection& kp_col, cv::Mat& descriptors, std::string parameter_file) {
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
	orb->compute(image, kp_col.keypoints, descriptors);
}

void sift(cv::Mat image, KeyPointCollection& kp_col, cv::Mat& descriptors, std::string parameter_file) {
	map<string, double> params = {{"n_features", 0},
								  {"n_octave_layers", 3},
								  {"contrast_threshold", 0.04},
								  {"edge_threshold", 10},
								  {"sigma", 1.6}};

	load_parameters(parameter_file, params);

	cv::Ptr<cv::Feature2D> sift = cv::xfeatures2d::SIFT::create(params["n_features"], params["n_octave_layers"],
			params["contrast_threshold"], params["edge_threshold"], params["sigma"]);
	sift->compute(image, kp_col.keypoints, descriptors);
}

void surf(cv::Mat image, KeyPointCollection& kp_col, cv::Mat& descriptors, std::string parameter_file) {
	map<string, double> params = {{"hessian_threshold", 100},
								  {"n_octaves", 4},
								  {"n_octave_layers", 3},
								  {"extended", 0}};
	bool upright = false;

	load_parameters(parameter_file, params);

	cv::Ptr<cv::Feature2D> surf = cv::xfeatures2d::SURF::create(params["hessian_threshold"], params["n_octaves"],
			params["n_octave_layers"], dtob(params["extended"]), upright);
	surf->compute(image, kp_col.keypoints, descriptors);
}

void usurf(cv::Mat image, KeyPointCollection& kp_col, cv::Mat& descriptors, std::string parameter_file) {
	map<string, double> params = {{"hessian_threshold", 100},
								  {"n_octaves", 4},
								  {"n_octave_layers", 3},
								  {"extended", 0}};
	bool upright = true;

	load_parameters(parameter_file, params);

	cv::Ptr<cv::Feature2D> surf = cv::xfeatures2d::SURF::create(params["hessian_threshold"], params["n_octaves"],
			params["n_octave_layers"], dtob(params["extended"]), upright);
	surf->compute(image, kp_col.keypoints, descriptors);
}

