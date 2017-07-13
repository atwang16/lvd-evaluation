/*
 * distancethresh.cpp
 *
 *  Created on: Jul 7, 2017
 *      Author: austin
 */

#include "utils.hpp"
#include <boost/filesystem.hpp>
#include <boost/algorithm/string.hpp>

//#define DEBUG

using namespace std;
using namespace cv;

int main(int argc, char *argv[]) {
	Mat img_1, img_2, desc_1, desc_2, homography;
	Mat kp_mat_1, kp_mat_2;
	string desc_name, results = "";
	vector<KeyPoint> kp_vec_1, kp_vec_2;
	float dist_ratio_thresh = 0.8f, kp_dist_thresh = 2.5f;
	int dist_metric, append = 1, verbose = 0;

	if(argc < 10) {
		cout << "Usage ./distances parameters_file desc_name desc_1 keypoint_1 desc2 keypoint2 homography dist_metric append [results]\n";
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
	desc_1 = parse_file(argv[3], ',', CV_8U);
	if(verbose) {
		cout << "Parsed descriptor 1.\n";
	}
	kp_mat_1 = parse_file(argv[4], ',', CV_32F);
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

	desc_2 = parse_file(argv[5], ',', CV_8U);
	if(verbose) {
		cout << "Parsed descriptor 2.\n";
	}
	kp_mat_2 = parse_file(argv[6], ',', CV_32F);
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
	homography = parse_file(argv[7], ' ', CV_32F);
	dist_metric = get_dist_metric(argv[8]);

	append = stoi(argv[9]);

	if(argc >= 11) {
		results = argv[10];
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

	// Use ground truth homography to determine positive versus negative matches
	vector<float> pos_dist, neg_dist;
	for(int i = 0; i < good_matches.size(); i++) {
		int kp_id_1 = good_matches[i][0].queryIdx;
		int kp_id_2 = good_matches[i][0].trainIdx;
		if(is_overlapping(kp_vec_1[kp_id_1], kp_vec_2[kp_id_2], homography, kp_dist_thresh)) {
			pos_dist.push_back(good_matches[i][0].distance);
			neg_dist.push_back(good_matches[i][1].distance);
		}
		else {
			neg_dist.push_back(good_matches[i][0].distance);
		}
	}

	vector<float> cor_dist;
	// Compute distances between correct correspondences
	for(int i = 0; i < kp_vec_1.size(); i++) {
		for(int j = 0; j < kp_vec_2.size(); j++) {
			if(is_overlapping(kp_vec_1[i], kp_vec_2[j], homography, kp_dist_thresh)) {
				cor_dist.push_back(norm(desc_1.row(i), desc_2.row(j), dist_metric));
				break;
			}
		}
	}

	// Save data to csv files
	ofstream f_pos, f_neg, f_cor;

	f_pos.open(results + desc_name + "_pos_dists.csv", ofstream::out | (ofstream::app * append));
	for(int i = 0; i < pos_dist.size(); i++) {
		f_pos << pos_dist[i] << "\n";
	}
	f_pos.close();

	f_neg.open(results + desc_name + "_neg_dists.csv", ofstream::out | (ofstream::app * append));
	for(int i = 0; i < neg_dist.size(); i++) {
		f_neg << neg_dist[i] << "\n";
	}
	f_neg.close();

	f_cor.open(results + desc_name + "_cor_dists.csv", ofstream::out | (ofstream::app * append));
	for(int i = 0; i < cor_dist.size(); i++) {
		f_cor << cor_dist[i] << "\n";
	}
	f_cor.close();

	return 0;
}
