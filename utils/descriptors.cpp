/*
 * detectors.cpp
 *
 *  Created on: Jul 24, 2017
 *      Author: austin
 */

//#include "descriptors.hpp"

#include "detectors.hpp"
#include "descriptors.hpp"

using namespace boost::filesystem;
using namespace std;
using namespace cv;


void akaze(cv::Mat image, vector<KeyPoint>& keypoints, vector<cv::Mat>& affine, cv::Mat& descriptors, std::string parameter_file) {
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
	akaze->detectAndCompute(image, cv::noArray(), keypoints, descriptors, false);
}

void brief(cv::Mat image, vector<KeyPoint>& keypoints, vector<cv::Mat>& affine, cv::Mat& descriptors, std::string parameter_file) {
	map<string, double> params = {{"bytes", 32},
								  {"use_orientation", 0}};

	load_parameters(parameter_file, params);

	cv::Ptr<cv::xfeatures2d::BriefDescriptorExtractor> brief =
			cv::xfeatures2d::BriefDescriptorExtractor::create(params["bytes"], dtob(params["use_orientation"]));
	brief->compute(image, keypoints, descriptors);
}

void sift(cv::Mat image, vector<KeyPoint>& keypoints, vector<cv::Mat>& affine, cv::Mat& descriptors, std::string parameter_file) {
	map<string, double> params = {{"n_features", 0},
								  {"n_octave_layers", 3},
								  {"contrast_threshold", 0.04},
								  {"edge_threshold", 10},
								  {"sigma", 1.6}};

	load_parameters(parameter_file, params);

	cv::Ptr<cv::Feature2D> sift = cv::xfeatures2d::SIFT::create(params["n_features"], params["n_octave_layers"],
			params["contrast_threshold"], params["edge_threshold"], params["sigma"]);
	sift->compute(image, keypoints, descriptors);
}

