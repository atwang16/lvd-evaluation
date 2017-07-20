/*
 * basetest.cpp
 *
 *  Created on: Jun 27, 2017
 *      Author: austin
 */

#include "utils.hpp"
#include <cmath>
#include <boost/filesystem.hpp>
#include <boost/algorithm/string.hpp>
#include <chrono>

//#define DEBUG

using namespace std;
using namespace cv;
using namespace std::chrono;

int main(int argc, char *argv[]) {
	Mat img_1, img_2, desc_1, desc_2, homography;
	Mat kp_mat_1, kp_mat_2;
	string dist_metric, desc_name, draw_results = "", stat_results = "", img_1_num, img_1_seq, img_2_num, img_2_seq;
	vector<KeyPoint> kp_vec_1, kp_vec_2;

	if(argc < 11) {
		cout << "Usage ./basetest parameters_file desc_name img_1 desc_1 keypoint_1 img_2 desc2 keypoint2 homography dist_metric [stat_results] [draw_results]\n";
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
	int verbose = 0;

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
	boost::split(img_1_path_split, argv[3], boost::is_any_of("/."));
	img_1_seq = img_1_path_split[img_1_path_split.size()-2].substr(0, 7);
	img_1_num = img_1_path_split[img_1_path_split.size()-2].substr(8, 3);
	desc_1 = parse_file(argv[4], ',', CV_8U);
	kp_mat_1 = parse_file(argv[5], ',', CV_32F);
	for(int i = 0; i < kp_mat_1.rows; i++) {
		KeyPoint kp_1 = KeyPoint();
		kp_1.pt.x = kp_mat_1.at<float>(i, 0);
		kp_1.pt.y = kp_mat_1.at<float>(i, 1);
		kp_1.size = kp_mat_1.at<float>(i, 2);
		kp_1.angle = kp_mat_1.at<float>(i, 3);
		kp_1.response = kp_mat_1.at<float>(i, 4);
		kp_1.octave = kp_mat_1.at<float>(i, 5);
		kp_vec_1.push_back(kp_1);
	}

	img_2 = imread(argv[6], CV_LOAD_IMAGE_COLOR);
	std::vector<std::string> img_2_path_split;
	boost::split(img_2_path_split, argv[6], boost::is_any_of("/,."));
	img_2_seq = img_2_path_split[img_2_path_split.size()-2].substr(0, 7);
	img_2_num = img_2_path_split[img_2_path_split.size()-2].substr(8, 3);
	desc_2 = parse_file(argv[7], ',', CV_8U);
	kp_mat_2 = parse_file(argv[8], ',', CV_32F);
	for(int i = 0; i < kp_mat_2.rows; i++) {
		KeyPoint kp_2 = KeyPoint();
		kp_2.pt.x = kp_mat_2.at<float>(i, 0);
		kp_2.pt.y = kp_mat_2.at<float>(i, 1);
		kp_2.size = kp_mat_2.at<float>(i, 2);
		kp_2.angle = kp_mat_2.at<float>(i, 3);
		kp_2.response = kp_mat_2.at<float>(i, 5);
		kp_2.octave = kp_mat_2.at<float>(i, 4);
		kp_vec_2.push_back(kp_2);
	}
	homography = parse_file(argv[9], ' ', CV_32F);
	dist_metric = argv[10];

	if(argc >= 12) {
		stat_results = argv[11];
	}
	if(argc >= 13) {
		draw_results = argv[12];
	}

	if(img_1_seq != img_2_seq) {
		cout << "Error: sequence codes do not match.\n";
		return 1;
	}

	// Match descriptors from 1 to 2 using nearest neighbor ratio
	high_resolution_clock::time_point start = high_resolution_clock::now();
	Ptr< BFMatcher > matcher = BFMatcher::create(get_dist_metric(dist_metric)); // no cross check
	vector< vector<DMatch> > nn_matches;
	matcher->knnMatch(desc_1, desc_2, nn_matches, 2);
	high_resolution_clock::time_point end = high_resolution_clock::now();
	long match_time = duration_cast<milliseconds>(end - start).count();
	long match_time_per_kp = duration_cast<microseconds>(end - start).count() / kp_vec_1.size();

	// Use the distance ratio to determine whether it is a "good" match
	vector<DMatch> good_matches;
	for(int i = 0; i < nn_matches.size(); i++) {
		if(nn_matches[i][0].distance < dist_ratio_thresh * nn_matches[i][1].distance) {
			good_matches.push_back(nn_matches[i][0]);
		}
	}

	// Use ground truth homography to check whether descriptors are actually matching, using associated keypoints
	vector<DMatch> correct_matches;
	for(int i = 0; i < good_matches.size(); i++) {
		int kp_id_1 = good_matches[i].queryIdx;
		int kp_id_2 = good_matches[i].trainIdx;
		if(is_overlapping(kp_vec_1[kp_id_1], kp_vec_2[kp_id_2], homography, kp_dist_thresh)) {
			correct_matches.push_back(good_matches[i]);
		}
	}

	// Count total number of correspondences
	int num_correspondences = 0;
	for(int i = 0; i < kp_vec_1.size(); i++) {
		for(int j = 0; j < kp_vec_2.size(); j++) {
			if(is_overlapping(kp_vec_1[i], kp_vec_2[j], homography, kp_dist_thresh)) {
				num_correspondences++;
				break;
			}
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
		cout << desc_name << " descriptor results:\n";
		cout << "-----------------------\n";
		cout << "Match ratio: " << match_ratio << "\n";
		cout << "Matching score: " << matching_score << "\n";
		cout << "Precision: " << precision << "\n";
		cout << "Recall: " << recall << "\n";
	}
	else {
		ofstream f;
		f.open(stat_results);
		f << "Descriptor:                          " << desc_name              << "\n";
		f << "Image 1:                             " << argv[3]                << "\n";
		f << "Image 2:                             " << argv[6]                << "\n";
		f << "Descriptor Size (bytes):             " << descriptor_size        << "\n";
		f                                                                      << "\n";
		f << "Number of Keypoints for Image 1:     " << kp_vec_1.size()        << "\n";
		f << "Number of Keypoints for Image 2:     " << kp_vec_2.size()        << "\n";
		f << "Number of Matches:                   " << good_matches.size()    << "\n";
		f << "Number of Correct Matches:           " << correct_matches.size() << "\n";
		f << "Number of Correspondences:           " << num_correspondences    << "\n";
		f                                                                      << "\n";
		f << "Match ratio:                         " << match_ratio            << "\n";
		f << "Matching score:                      " << matching_score         << "\n";
		f << "Precision:                           " << precision              << "\n";
		f << "Recall:                              " << recall                 << "\n";
		f << "Matching time (ms):                  " << match_time             << "\n";
		f << "Matching time per keypoint (mus):    " << match_time_per_kp      << "\n";
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
