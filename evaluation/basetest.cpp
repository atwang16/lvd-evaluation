/*
 * basetest.cpp
 *
 *  Created on: Jun 27, 2017
 *      Author: austin
 */

#include "utils.hpp"
#include <opencv2/opencv.hpp>
#include <boost/filesystem.hpp>
#include <boost/algorithm/string.hpp>
#include <chrono>
#include <cmath>

#define PATH_DELIMITER "/"
#define BOOST_PATH_DELIMITER boost::is_any_of(PATH_DELIMITER)

using namespace std;
using namespace cv;
using namespace std::chrono;

int main(int argc, char *argv[]) {
	Mat img_1, img_2, desc_1, desc_2, homography;
	Mat kp_mat_1, kp_mat_2;
	string dist_metric, desc_name, draw_results = "", stat_results = "";
	string img_1_path, img_1_name, img_2_path, img_2_name;
	int num_correspondences;
	vector<KeyPoint> kp_vec_1, kp_vec_2;
	vector< set<int> > correspondences;

	if(argc < 12) {
		cout << "Usage ./basetest parameters_file desc_name img_1 desc_1 keypoint_1 img_2 desc_2 keypoint_2 homography dist_metric num_correspondences [-s stat_results] [-d draw_results]\n";
		return 1;
	}

	// Load parameters for base test
	ifstream params(argv[1]);
	string line, var, value;
	std::vector<std::string> line_split;

	float dist_ratio_thresh = 0.8f;
	float kp_dist_thresh = 2.5f;
	int cap_correct_displayed = 1;
	int nb_kp_to_display = 50;

	while(getline(params, line)) {
		boost::split(line_split, line, boost::is_any_of("="));
		var = line_split[0];
		value = line_split[1];
		if(var == "DIST_RATIO_THRESH") {
			dist_ratio_thresh = stof(value);
		}
		else if(var == "KP_DIST_THRESH") {
			kp_dist_thresh = stof(value);
		}
		else if(var == "CAP_CORRECT_DISPLAYED") {
			cap_correct_displayed = stof(value);
		}
		else if(var == "NB_KP_TO_DISPLAY") {
			nb_kp_to_display = stof(value);
		}
	}

	// Parse remaining arguments
	vector<string> img_1_split, img_2_split;
	desc_name = argv[2];
	img_1_path = argv[3];
	boost::split(img_1_split, img_1_path, BOOST_PATH_DELIMITER);
	img_1_name = img_1_split.back();
	img_1 = imread(img_1_path, CV_LOAD_IMAGE_COLOR);
	desc_1 = parse_file(argv[4], ',');
	kp_mat_1 = parse_file(argv[5], ',');
	for(int i = 0; i < kp_mat_1.rows; i++) {
		KeyPoint kp_1 = KeyPoint();
		kp_1.pt.x = kp_mat_1.at<float>(i, 0);
		kp_1.pt.y = kp_mat_1.at<float>(i, 1);
		kp_vec_1.push_back(kp_1);
	}

	img_2_path = argv[6];
	boost::split(img_2_split, img_2_path, BOOST_PATH_DELIMITER);
	img_2_name = img_2_split.back();
	img_2 = imread(img_2_path, CV_LOAD_IMAGE_COLOR);
	desc_2 = parse_file(argv[7], ',');
	kp_mat_2 = parse_file(argv[8], ',');
	for(int i = 0; i < kp_mat_2.rows; i++) {
		KeyPoint kp_2 = KeyPoint();
		kp_2.pt.x = kp_mat_2.at<float>(i, 0);
		kp_2.pt.y = kp_mat_2.at<float>(i, 1);
		kp_vec_2.push_back(kp_2);
	}
	homography = parse_file(argv[9], ' ', CV_32F);
	dist_metric = argv[10];

	num_correspondences = stoi(argv[11]);

	for(int arg_ind = 12; arg_ind < argc; arg_ind++) {
		if(strcmp(argv[arg_ind], "-s") == 0) {
			if(++arg_ind < argc) {
				stat_results = argv[arg_ind];
			}
		}
		else if(strcmp(argv[arg_ind], "-d") == 0) {
			if(++arg_ind < argc) {
				draw_results = argv[arg_ind];
			}
		}
	}

	// Match descriptors from 1 to 2 using nearest neighbor ratio
	high_resolution_clock::time_point start = high_resolution_clock::now();
	Ptr< BFMatcher > matcher = BFMatcher::create(get_dist_metric(dist_metric)); // no cross check
	vector< vector<DMatch> > nn_matches;
	matcher->knnMatch(desc_1, desc_2, nn_matches, 2);

	// Use the distance ratio to determine whether it is a "good" match
	vector<DMatch> good_matches;
	for(int i = 0; i < nn_matches.size(); i++) {
		if(nn_matches[i][0].distance < dist_ratio_thresh * nn_matches[i][1].distance) {
			good_matches.push_back(nn_matches[i][0]);
		}
	}
	high_resolution_clock::time_point end = high_resolution_clock::now();
	long match_time = duration_cast<milliseconds>(end - start).count();

	// Use ground truth homography to check whether descriptors are actually matching, using associated keypoints
	vector<DMatch> correct_matches;
	for(int i = 0; i < good_matches.size(); i++) {
		int kp_id_1 = good_matches[i].queryIdx;
		int kp_id_2 = good_matches[i].trainIdx;
		if(is_overlapping(kp_vec_1[kp_id_1], kp_vec_2[kp_id_2], homography, kp_dist_thresh)) {
			correct_matches.push_back(good_matches[i]);
		}
	}

	// Use data from last step to build metrics
	float match_ratio = (float)good_matches.size() / kp_vec_1.size();
	float precision = (float)correct_matches.size() / good_matches.size();
	float matching_score = match_ratio * precision;
	float recall = (float)correct_matches.size() / num_correspondences;
	int descriptor_size = desc_1.cols * desc_1.elemSize1();

	// Export metrics
	if(stat_results == "") { // no directory specified for exporting results; print to console
		cout << "Descriptor:                          " << desc_name              << "\n";
		cout << "Image 1:                             " << img_1_path             << "\n";
		cout << "Image 2:                             " << img_2_path             << "\n";
		cout << "Descriptor Size (bytes):             " << descriptor_size        << "\n";
		cout                                                                      << "\n";
		cout << "Number of Keypoints for Image 1:     " << kp_vec_1.size()        << "\n";
		cout << "Number of Keypoints for Image 2:     " << kp_vec_2.size()        << "\n";
		cout << "Number of Matches:                   " << good_matches.size()    << "\n";
		cout << "Number of Correct Matches:           " << correct_matches.size() << "\n";
		cout << "Number of Correspondences:           " << num_correspondences    << "\n";
		cout                                                                      << "\n";
		cout << "Match ratio:                         " << match_ratio            << "\n";
		cout << "Matching score:                      " << matching_score         << "\n";
		cout << "Precision:                           " << precision              << "\n";
		cout << "Recall:                              " << recall                 << "\n";
		cout << "Matching time (ms):                  " << match_time             << "\n";
	}
	else {
		ofstream f;
		f.open(stat_results, ofstream::out | ofstream::app);
		f << img_1_name             << ",";
		f << img_2_name             << ",";
		f << kp_vec_1.size()        << ",";
		f << kp_vec_2.size()        << ",";
		f << good_matches.size()    << ",";
		f << correct_matches.size() << ",";
		f << num_correspondences    << ",";
		f << match_ratio            << ",";
		f << matching_score         << ",";
		f << precision              << ",";
		f << recall                 << ",";
		f << match_time             << "\n";
		f.close();
	}


	// Extract the first nb_kp_to_display matches to show in image
	vector< DMatch >::const_iterator first = correct_matches.begin();
	vector< DMatch >::const_iterator last;
	if(cap_correct_displayed && correct_matches.size() > nb_kp_to_display) {
		last = correct_matches.begin() + nb_kp_to_display;
	}
	else {
		last = correct_matches.end();
	}
	vector< DMatch > display_matches(first, last);;

	// Draw matches
	if(draw_results != "") {
		Mat res;
		drawMatches(img_1, kp_vec_1, img_2, kp_vec_2, display_matches, res, Scalar::all(-1), Scalar::all(-1),
				vector< char >(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
		imwrite(draw_results, res);
	}

	return 0;
}
