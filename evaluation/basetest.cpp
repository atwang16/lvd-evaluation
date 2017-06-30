/*
 * basetest.cpp
 *
 *  Created on: Jun 27, 2017
 *      Author: austin
 */

#include <opencv2/opencv.hpp>
#include <string>
#include <iostream>
#include <sstream>
#include <fstream>
#include <cmath>

#define FIR_NEAR 0
#define SEC_NEAR 1
#define DIST_RATIO_THRESH 0.8
#define OVERLAP_THRESH 0.5

cv::Mat parse_file(std::string fname, const char delimiter);
int get_dist_metric(std::string metric);
bool is_overlapping(cv::KeyPoint kp_1, cv::KeyPoint kp_2, float threshold);

int main(int argc, char *argv[]) {
//	if(argc < 3) {
//		std::cout << "Usage ./basetest" << " descriptor_name descriptor_data_1 descriptor_data_2 ground_truths" << "\n";
//		return -1;
//	}
	std::string descr_name = argv[1];
	std::string path_to_keypoint_1 = argv[2];
	std::string path_to_keypoint_2 = argv[3];
	std::string path_to_descriptor_1 = argv[4];
	std::string path_to_descriptor_2 = argv[5];
	std::string path_to_ground_truth = argv[6];
	std::string dist_metric = argv[7];

	// Load descriptors and keypoints from files
	cv::Mat kp_mat_1 = parse_file(path_to_keypoint_1, ','); // matrix of keypoints for image 1
	cv::Mat kp_mat_2 = parse_file(path_to_keypoint_2, ','); // matrix of keypoints for image 2
	cv::Mat descr_1 = parse_file(path_to_descriptor_1, ','); // matrix of descriptors for image 1
	cv::Mat descr_2 = parse_file(path_to_descriptor_2, ','); // matrix of descriptors for image 2
	cv::Mat homography = parse_file(path_to_ground_truth, ' '); // matrix of ground truth homography from image 1 to 2

	// Match descriptors from 1 to 2 using nearest neighbor
	cv::Ptr< cv::BFMatcher > matcher = cv::BFMatcher::create(cv::BFMatcher::normType=get_dist_metric(dist_metric));
	std::vector< std::vector<cv::DMatch> > matches;
	matcher->knnMatch(descr_1, descr_2, matches, 2);

	// Use the distance ratio to determine whether it is a "good" match
	std::vector<cv::DMatch> good_matches;
	for(int i = 0; i < descr_1.rows; i++) {
		if(matches[i][FIR_NEAR].distance/matches[i][SEC_NEAR] < DIST_RATIO_THRESH) {
			good_matches.push_back(matches[i][FIR_NEAR]);
		}
	}

	// Use ground truth homography to check whether descriptors are actually matching, using associated keypoints
	std::vector<cv::DMatch> correct_matches;
	for(int i = 0; i < good_matches.size(); i++) {
		int kp_id_1 = good_matches[i].queryIdx;
		int kp_id_2 = good_matches[i].trainIdx;
		cv::KeyPoint kp_1 = cv::KeyPoint(kp_mat_1[kp_id_1][0], kp_mat_1[kp_id_1][1], kp_mat_1[kp_id_1][2],
				kp_mat_1[kp_id_1][3], kp_mat_1[kp_id_1][4]);
		cv::KeyPoint kp_2 = cv::KeyPoint(kp_mat_2[kp_id_2][0], kp_mat_2[kp_id_2][1], kp_mat_2[kp_id_2][2],
				kp_mat_2[kp_id_2][3], kp_mat_2[kp_id_2][4]);
		if(is_overlapping(kp_1, kp_2, OVERLAP_THRESH)) {
			correct_matches.push_back(good_matches[i]);
		}
	}

	// Count total number of correspondences
	int num_correspondences = 0;
	for(int i = 0; i < kp_mat_1.rows; i++) {
		cv::KeyPoint kp_1 = cv::KeyPoint(kp_mat_1[i][0], kp_mat_1[i][1], kp_mat_1[i][2], kp_mat_1[i][3], kp_mat_1[i][4]);
		for(int j = 0; j < kp_mat_2.rows; j++) {
			cv::KeyPoint kp_2 = cv::KeyPoint(kp_mat_2[j][0], kp_mat_2[j][1], kp_mat_2[j][2], kp_mat_2[j][3], kp_mat_2[j][4]);
			if(is_overlapping(kp_1, kp_2, OVERLAP_THRESH)) {
				num_correspondences++;
			}
		}
	}

	// Use data from last step to build metrics
	float match_ratio = good_matches.size() / descr_1.size();   // Since we are not cross checking, each descriptor in image 1 is
																// matched to one in image 2, thus we count the number of features
																// detected as the number of descriptors found for image 1.
	float precision = correct_matches.size() / good_matches.size();
	float matching_score = match_ratio * precision;
	float recall = correct_matches.size() / num_correspondences;

	// Export metrics
	std::cout << "Match ratio: " << match_ratio << "\n";
	std::cout << "Matching score: " << matching_score << "\n";
	std::cout << "Precision: " << precision << "\n";
	std::cout << "Recall: " << recall << "\n";

	return 0;
}

cv::Mat parse_file(std::string fname, const char delimiter) { // TODO: modify to adjust for type
	std::ifstream inputfile(fname);
	std::string current_line;
	std::vector<std::vector<int>> all_data;

	// read each line
	while(std::getline(inputfile, current_line)) {
		std::vector<int> values;
		std::stringstream temp(current_line);
		std::string single_value;

		// Read each value with delimiter
		while(std::getline(temp,single_value, delimiter)) {
	        values.push_back(atoi(single_value.c_str()));
		}
		all_data.push_back(values);
	}

	// Place data in OpenCV matrix
	cv::Mat vect = cv::Mat::zeros((int)all_data.size(), (int)all_data[0].size(), CV_8UC1);
	for(int row = 0; row < (int)all_data.size(); row++) {
	   for(int col = 0; col < (int)all_data[0].size(); col++) {
	      vect.at<uchar>(row,col) = all_data[row][col];
	   }
	}
	return vect;
}

int get_dist_metric(std::string metric) {
	switch(metric) {
	case "L2":
		return cv::NORM_L2;
	case "L1":
		return cv::NORM_L1;
	case "HAMMING":
		return cv::NORM_HAMMING;
	case "HAMMING2":
		return cv::NORM_HAMMING2;
	default:
		return cv::NORM_L2;
	}
}

bool is_overlapping(cv::KeyPoint kp_1, cv::KeyPoint kp_2, float threshold) {

}

cv::Mat get_affine_transformation(int x, int y, int scale_factor, cv::Mat hom) {
	cv::Mat2f pos = cv::Mat2f::zeros(3, 1);
	pos[0] = x;
	pos[1] = y;
	pos[2] = 1;

	float fxdx = hom[0][0]/(pos.dot(hom.row(2))) - hom[2][0](pos.dot(hom.row(0)))/std::pow(pos.dot(hom.row(2)), 2);
	float fxdy = hom[0][1]/(pos.dot(hom.row(2))) - hom[2][1](pos.dot(hom.row(0)))/std::pow(pos.dot(hom.row(2)), 2);
	float fydx = hom[1][0]/(pos.dot(hom.row(2))) - hom[2][0](pos.dot(hom.row(1)))/std::pow(pos.dot(hom.row(2)), 2);
	float fydy = hom[1][1]/(pos.dot(hom.row(2))) - hom[2][1](pos.dot(hom.row(1)))/std::pow(pos.dot(hom.row(2)), 2);

	cv::Mat2f aff = cv::Mat2f::zeros(2, 2);
	aff[0][0] = fxdx;
	aff[0][1] = fxdy;
	aff[1][0] = fydx;
	aff[1][1] = fydy;

	return aff;
}
