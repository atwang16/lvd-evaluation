/*
 * basetest.cpp
 *
 *  Created on: Jun 27, 2017
 *      Author: austin
 */

#include <opencv2/opencv.hpp>
#include <cstdint>
#include <string>
#include <iostream>
#include <sstream>
#include <fstream>
#include <cmath>
#include <boost/filesystem.hpp>
#include <boost/algorithm/string.hpp>

//#define DEBUG

using namespace std;
using namespace cv;

Mat parse_file(string fname, char delimiter, int type);
int get_dist_metric(string metric);
bool is_overlapping(KeyPoint kp_1, KeyPoint kp_2, Mat hom, float threshold);

int main(int argc, char *argv[]) {
	Mat img_1, img_2, desc_1, desc_2, homography;
	Mat kp_mat_1, kp_mat_2;
	string desc_name, results = "", img_1_name, img_2_name;
	vector<KeyPoint> kp_vec_1, kp_vec_2;
	float dist_ratio_thresh = 0.8f, kp_dist_thresh = 2.5f;
	int dist_metric, append = 1, verbose = 0;

	if(argc < 12) {
		cout << "Usage ./distances parameters_file desc_name img_1 desc_1 keypoint_1 img_2 desc2 keypoint2 homography dist_metric append [results]\n";
		return -1;
	}

	// Load parameters for base test
	ifstream params(argv[1]);
	string line, var, value;
	std::vector<std::string> line_split;

	while(getline(params, line)) {
		boost::split(line_split, line, boost::is_any_of("="));
		var = line_split[0];
		value = line_split.back();
		if(var == "DIST_RATIO_THRESH") {
			dist_ratio_thresh = stof(value);
		}
		else if(var == "KP_DIST_THRESH") {
			kp_dist_thresh = stof(value);
		}
		else if(var == "VERBOSE") {
			verbose = stof(value);
		}
	}

	if(verbose) {
		cout << "Starting execution.\n";
	}

	// Parse remaining arguments
	desc_name = argv[2];
	img_1 = imread(argv[3], CV_LOAD_IMAGE_COLOR);
	std::vector<std::string> img_1_path_split;
	boost::split(img_1_path_split, argv[3], boost::is_any_of("/,."));
	img_1_name = img_1_path_split[img_1_path_split.size()-2];
	if(verbose) {
		cout << "Read image 1.\n";
	}
	desc_1 = parse_file(argv[4], ',', CV_8U);
	if(verbose) {
		cout << "Parsed descriptor 1.\n";
	}
	kp_mat_1 = parse_file(argv[5], ',', CV_32F);
	if(verbose) {
		cout << "Parsed " << kp_mat_1.rows << " keypoints for image 1.\n";
	}
	for(int i = 0; i < kp_mat_1.rows; i++) {
		KeyPoint kp_1 = KeyPoint();
		kp_1.pt.x = kp_mat_1.at<float>(i, 0);
		kp_1.pt.y = kp_mat_1.at<float>(i, 1);
		kp_1.size = kp_mat_1.at<float>(i, 2);
		kp_1.angle = kp_mat_1.at<float>(i, 3);
		kp_1.octave = kp_mat_1.at<float>(i, 4);
		kp_1.response = kp_mat_1.at<float>(i, 5);
		kp_vec_1.push_back(kp_1);
	}

	img_2 = imread(argv[6], CV_LOAD_IMAGE_COLOR);
	std::vector<std::string> img_2_path_split;
	boost::split(img_2_path_split, argv[6], boost::is_any_of("/,."));
	img_2_name = img_2_path_split[img_2_path_split.size()-2];
	if(verbose) {
		cout << "Read image 2.\n";
	}
	desc_2 = parse_file(argv[7], ',', CV_8U);
	if(verbose) {
		cout << "Parsed descriptor 2.\n";
	}
	kp_mat_2 = parse_file(argv[8], ',', CV_32F);
	if(verbose) {
		cout << "Parsed " << kp_mat_2.rows << " keypoints for image 2.\n";
	}
	for(int i = 0; i < kp_mat_2.rows; i++) {
		KeyPoint kp_2 = KeyPoint();
		kp_2.pt.x = kp_mat_2.at<float>(i, 0);
		kp_2.pt.y = kp_mat_2.at<float>(i, 1);
		kp_2.size = kp_mat_2.at<float>(i, 2);
		kp_2.angle = kp_mat_2.at<float>(i, 3);
		kp_2.octave = kp_mat_2.at<float>(i, 4);
		kp_2.response = kp_mat_2.at<float>(i, 5);
		kp_vec_2.push_back(kp_2);
	}
	homography = parse_file(argv[9], ' ', CV_32F);
	dist_metric = get_dist_metric(argv[10]);

	append = (argv[11][0] == '1');

	if(argc >= 13) {
		results = argv[12];
	}

	if(verbose) {
		cout << "Processed input.\n";
	}

	// Match descriptors from 1 to 2 using nearest neighbor ratio
	Ptr< BFMatcher > matcher = BFMatcher::create(dist_metric); // no cross check
	vector< vector<DMatch> > nn_matches;
	matcher->knnMatch(desc_1, desc_2, nn_matches, 2);

	if(verbose) {
		cout << "Matched descriptors.\n";
	}

	// Use the distance ratio to determine whether it is a "good" match
	vector< vector<DMatch> > good_matches;
	for(int i = 0; i < nn_matches.size(); i++) {
		if(nn_matches[i][0].distance < dist_ratio_thresh * nn_matches[i][1].distance) {
			good_matches.push_back(nn_matches[i]);
		}
	}

	// Use ground truth homography to check whether descriptors are actually matching, using associated keypoints
	vector<DMatch> correct_matches;
	vector<float> good_match_dist, bad_match_dist;
	for(int i = 0; i < good_matches.size(); i++) {
		int kp_id_1 = good_matches[i][0].queryIdx;
		int kp_id_2 = good_matches[i][0].trainIdx;
		if(is_overlapping(kp_vec_1[kp_id_1], kp_vec_2[kp_id_2], homography, kp_dist_thresh)) {
			good_match_dist.push_back(good_matches[i][0].distance);
			bad_match_dist.push_back(good_matches[i][1].distance);
		}
		else {
			bad_match_dist.push_back(good_matches[i][0].distance);
		}
	}

	vector<float> corr_match_dist;
	// Count total number of correspondences
	for(int i = 0; i < kp_vec_1.size(); i++) {
		for(int j = 0; j < kp_vec_2.size(); j++) {
			if(is_overlapping(kp_vec_1[i], kp_vec_2[j], homography, kp_dist_thresh)) {
				corr_match_dist.push_back(norm(desc_1.row(i), desc_2.row(j), dist_metric));
				break;
			}
		}
	}

	ofstream f_goodmatches, f_badmatches, f_corrmatches;

	f_goodmatches.open(results + desc_name + "_goodmatches_distthresh.csv", ofstream::out | (ofstream::app * append));
	for(int i = 0; i < good_match_dist.size(); i++) {
		f_goodmatches << good_match_dist[i] << "\n";
	}
	f_goodmatches.close();

	f_badmatches.open(results + desc_name + "_badmatches_distthresh.csv", ofstream::out | (ofstream::app * append));
	for(int i = 0; i < bad_match_dist.size(); i++) {
		f_badmatches << bad_match_dist[i] << "\n";
	}
	f_badmatches.close();

	f_corrmatches.open(results + desc_name + "_corrmatches_distthresh.csv", ofstream::out | (ofstream::app * append));
	for(int i = 0; i < corr_match_dist.size(); i++) {
		f_corrmatches << corr_match_dist[i] << "\n";
	}
	f_corrmatches.close();

	return 0;
}

Mat parse_file(string fname, char delimiter, int type) {
	ifstream inputfile(fname);
	string current_line;

	if(type != CV_8U && type != CV_32F) {
		cout << "Error: invalid type passed to parse_file. Default float assumed.\n";
		type = CV_32F;
	}

	if(type == CV_32F) {
		vector< vector<float> > all_data;

		// read each line
		while(getline(inputfile, current_line)) {
			if(current_line != "") {
				vector<float> values;
				stringstream str_stream(current_line);
				string single_value;

				// Read each value with delimiter
				while(getline(str_stream,single_value, delimiter)) {
					if(single_value != "") {
						values.push_back(atof(single_value.c_str()));
					}
				}
				all_data.push_back(values);
			}
		}

		// Place data in OpenCV matrix
		Mat vect = Mat::zeros((int)all_data.size(), (int)all_data[0].size(), CV_32F);
		for(int row = 0; row < vect.rows; row++) {
		   for(int col = 0; col < vect.cols; col++) {
			  vect.at<float>(row, col) = all_data[row][col];
		   }
		}
		return vect;
	}
	else { // CV_8U
		vector< vector<uint8_t> > all_data;

		// read each line
		while(getline(inputfile, current_line)) {
			if(current_line != "") {
				vector<uint8_t> values;
				stringstream str_stream(current_line);
				string single_value;

				// Read each value with delimiter
				while(getline(str_stream,single_value, delimiter)) {
					if(single_value != "") {
						values.push_back(atoi(single_value.c_str()));
					}
				}
				all_data.push_back(values);
			}
		}

		// Place data in OpenCV matrix
		Mat vect = Mat::zeros((int)all_data.size(), (int)all_data[0].size(), CV_8U);
		for(int row = 0; row < vect.rows; row++) {
		   for(int col = 0; col < vect.cols; col++) {
			  vect.at<uint8_t>(row, col) = all_data[row][col];
		   }
		}
		return vect;
	}
}

int get_dist_metric(string metric) {
	if(metric == "L2") {
		return NORM_L2;
	}
	else if(metric == "L1") {
		return NORM_L1;
	}
	else if(metric == "HAMMING") {
		return NORM_HAMMING;
	}
	else if(metric == "HAMMING2") {
		return NORM_HAMMING2;
	}
	else { // default
		return NORM_L2;
	}
}

bool is_overlapping(KeyPoint kp_1, KeyPoint kp_2, Mat hom, float threshold) {
	Mat hom_coord_vec = Mat::ones(3, 1, CV_32FC1);
	hom_coord_vec.at<float>(0) = kp_1.pt.x;
	hom_coord_vec.at<float>(1) = kp_1.pt.y;
	Mat proj_kp_1_2 =  hom * hom_coord_vec;
	proj_kp_1_2 /= proj_kp_1_2.at<float>(2);
	float dist = sqrt(pow(kp_2.pt.x - proj_kp_1_2.at<float>(0), 2) + pow(kp_2.pt.y - proj_kp_1_2.at<float>(1), 2));
	return dist < threshold;
}
