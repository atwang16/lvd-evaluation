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
#define KP_DIST_THRESH 2.5f

using namespace std;
using namespace cv;

Mat parse_file(string fname, char delimiter);
int get_dist_metric(string metric);
bool is_overlapping(KeyPoint kp_1, KeyPoint kp_2, Mat hom, float threshold);

int main(int argc, char *argv[]) {
	Mat img_1, img_2, desc_1, desc_2, homography;
	Mat kp_mat_1, kp_mat_2;
	string dist_metric, desc_name;
	vector<KeyPoint> kp_vec_1, kp_vec_2;
	bool draw_matches;

	cout << "Starting execution.\n";
	if(argc < 11) {
		cout << "Usage ./basetest desc_name img_1 desc_1 keypoint_1 img_2 desc2 keypoint2 homography dist_metric display_matches\n";
		return -1;
	}
	else {
		cout << "Output received.\n";
	}

	desc_name = argv[1];
	img_1 = imread(argv[2], CV_LOAD_IMAGE_COLOR);
	cout << "Read image 1.\n";
	desc_1 = parse_file(argv[3], ',');
	cout << "Parsed descriptor 1.\n";
	kp_mat_1 = parse_file(argv[4], ',');
	cout << "Parsed keypoints 1.\n";
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
	img_2 = imread(argv[5], CV_LOAD_IMAGE_COLOR);
	cout << "Read image 2.\n";
	desc_2 = parse_file(argv[6], ',');
	cout << "Parsed descriptor 2.\n";
	kp_mat_2 = parse_file(argv[7], ',');
	cout << "Parsed keypoints 2.\n";
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
	homography = parse_file(argv[8], ' ');
	dist_metric = argv[9];
	draw_matches = (argv[10][0] == '1');

	cout << "Processed input.\n";

	// Match descriptors from 1 to 2 using nearest neighbor
	Ptr< BFMatcher > matcher = BFMatcher::create(get_dist_metric(dist_metric));
	vector< vector<DMatch> > matches;
	matcher->knnMatch(desc_1, desc_2, matches, 2);

	cout << "Matched descriptors.\n";

	// Use the distance ratio to determine whether it is a "good" match
	vector<DMatch> good_matches;
	for(int i = 0; i < matches.size(); i++) {
		if(matches[i][FIR_NEAR].distance < DIST_RATIO_THRESH * matches[i][SEC_NEAR].distance) {
			good_matches.push_back(matches[i][FIR_NEAR]); // we only need to save the first match
		}
	}

	cout << "Good matches determined.\n";

	// Use ground truth homography to check whether descriptors are actually matching, using associated keypoints
	vector<DMatch> correct_matches;
	vector<KeyPoint> matched_kp_1, matched_kp_2;
	for(int i = 0; i < good_matches.size(); i++) {
		int kp_id_1 = good_matches[i].queryIdx;
		int kp_id_2 = good_matches[i].trainIdx;
		if(is_overlapping(kp_vec_1[kp_id_1], kp_vec_2[kp_id_2], homography, KP_DIST_THRESH)) {
			correct_matches.push_back(good_matches[i]);
			matched_kp_1.push_back(kp_vec_1[kp_id_1]);
			matched_kp_2.push_back(kp_vec_2[kp_id_2]);
		}
	}

	cout << "Evaluated matches using ground truths.\n";

	// Count total number of correspondences
	int num_correspondences = 0;
	for(int i = 0; i < kp_vec_1.size(); i++) {
		for(int j = 0; j < kp_vec_2.size(); j++) {
			if(is_overlapping(kp_vec_1[i], kp_vec_2[j], homography, KP_DIST_THRESH)) {
				num_correspondences++;
				break;
			}
		}
	}

	// Use data from last step to build metrics
	float match_ratio = good_matches.size() / desc_1.rows;   // Since we are not cross checking, each descriptor in image 1 is
															 // matched to one in image 2, thus we count the number of features
															 // detected as the number of descriptors found for image 1.
	float precision = correct_matches.size() / good_matches.size();
	float matching_score = match_ratio * precision;
	float recall = correct_matches.size() / num_correspondences;

	// TODO: make it export to file instead of printing only to console
	// Export metrics
	cout << desc_name << " descriptor results:\n";
	cout << "-----------------------\n";
	cout << "Match ratio: " << match_ratio << "\n";
	cout << "Matching score: " << matching_score << "\n";
	cout << "Precision: " << precision << "\n";
	cout << "Recall: " << recall << "\n";

	// TODO: Export image to file
	// Draw matches
	if(draw_matches) {
		Mat res;
		drawMatches(img_1, matched_kp_1, img_2, matched_kp_2, correct_matches, res);
//		imwrite("../../samples/data/latch_res.png", res);
	}

	return 0;
}

Mat parse_file(string fname, char delimiter) { // TODO: modify to adjust for type
	ifstream inputfile(fname);
	string current_line;
	vector< vector<float> > all_data;

	// TODO: Check for empty lines and skip over them
	// read each line
	while(getline(inputfile, current_line)) {
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

	// Place data in OpenCV matrix
	Mat vect = Mat::zeros((int)all_data.size(), (int)all_data[0].size(), CV_32F);
//	cout << all_data.size() << "\n";
//	cout << vect.rows << "\n";
	for(int row = 0; row < vect.rows; row++) {
	   for(int col = 0; col < vect.cols; col++) {
	      vect.at<float>(row, col) = all_data[row][col];
	   }
	}
	return vect;
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
	hom_coord_vec.at<float>(0, 0) = kp_1.pt.x;
	hom_coord_vec.at<float>(1, 0) = kp_1.pt.y;
	cout << hom_coord_vec.depth() << " " << hom_coord_vec.channels() << "\n";
	cout << hom.depth() << " " << hom.channels() << "\n";
	Mat proj_kp_1_2 =  hom * hom_coord_vec;
	float dist = sqrt(pow(kp_1.pt.x - proj_kp_1_2.at<float>(0, 0), 2) + pow(kp_1.pt.y - proj_kp_1_2.at<float>(1, 0), 2));
	return dist < threshold;
}

//Mat get_affine_transformation(int x, int y, int scale_factor, Mat hom) {
//	Mat2f pos = Mat2f::zeros(3, 1);
//	pos[0] = x;
//	pos[1] = y;
//	pos[2] = 1;
//
//	float fxdx = hom[0][0]/(pos.dot(hom.row(2))) - hom[2][0](pos.dot(hom.row(0)))/pow(pos.dot(hom.row(2)), 2);
//	float fxdy = hom[0][1]/(pos.dot(hom.row(2))) - hom[2][1](pos.dot(hom.row(0)))/pow(pos.dot(hom.row(2)), 2);
//	float fydx = hom[1][0]/(pos.dot(hom.row(2))) - hom[2][0](pos.dot(hom.row(1)))/pow(pos.dot(hom.row(2)), 2);
//	float fydy = hom[1][1]/(pos.dot(hom.row(2))) - hom[2][1](pos.dot(hom.row(1)))/pow(pos.dot(hom.row(2)), 2);
//
//	Mat2f aff = Mat2f::zeros(2, 2);
//	aff[0][0] = fxdx;
//	aff[0][1] = fxdy;
//	aff[1][0] = fydx;
//	aff[1][1] = fydy;
//
//	return aff;
//}
