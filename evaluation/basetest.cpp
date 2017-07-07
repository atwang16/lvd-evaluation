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
#include <chrono>

//#define DEBUG

using namespace std;
using namespace cv;
using namespace std::chrono;

Mat parse_file(string fname, char delimiter, int type);
int get_dist_metric(string metric);
bool is_overlapping(KeyPoint kp_1, KeyPoint kp_2, Mat hom, float threshold);

int main(int argc, char *argv[]) {
	Mat img_1, img_2, desc_1, desc_2, homography;
	Mat kp_mat_1, kp_mat_2;
	string dist_metric, desc_name, results = "", img_1_name, img_2_name;
	vector<KeyPoint> kp_vec_1, kp_vec_2;
	float dist_ratio_thresh = 0.8f, kp_dist_thresh = 2.5f;
	int nb_kp_to_display = 50, cap_correct_displayed = 1, verbose = 0;

	if(argc < 11) {
		cout << "Usage ./basetest parameters_file desc_name img_1 desc_1 keypoint_1 img_2 desc2 keypoint2 homography dist_metric [results]\n";
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
	dist_metric = argv[10];

	if(argc >= 12) {
		results = argv[11];
	}

	if(verbose) {
		cout << "Processed input.\n";
	}

	// Match descriptors from 1 to 2 using nearest neighbor ratio
	high_resolution_clock::time_point start = high_resolution_clock::now();
	Ptr< BFMatcher > matcher = BFMatcher::create(get_dist_metric(dist_metric)); // no cross check
	vector< vector<DMatch> > nn_matches;
	matcher->knnMatch(desc_1, desc_2, nn_matches, 2);
	high_resolution_clock::time_point end = high_resolution_clock::now();
	long match_time = duration_cast<milliseconds>(end - start).count();
	long match_time_per_kp = duration_cast<microseconds>(end - start).count() / kp_vec_1.size();

	if(verbose) {
		cout << "Matched descriptors.\n";
	}

#ifdef DEBUG
	vector<float> match_distances, match_indices;
	vector<string> is_good_match;
#endif

	// Use the distance ratio to determine whether it is a "good" match
	vector<DMatch> good_matches;
	for(int i = 0; i < nn_matches.size(); i++) {
#ifdef DEBUG
		match_distances.push_back(matches[i][FIR_NEAR].distance);
		match_indices.push_back(matches[i][FIR_NEAR].trainIdx);
#endif
		if(nn_matches[i][0].distance < dist_ratio_thresh * nn_matches[i][1].distance) {
			good_matches.push_back(nn_matches[i][0]);
#ifdef DEBUG
			is_good_match.push_back("good");
#endif
		}
#ifdef DEBUG
		else {
			is_good_match.push_back("no match");
		}
#endif
	}

	if(verbose) {
		cout << good_matches.size() << " good matches determined.\n";
	}

//	// Generate all projected keypoints
//	vector< KeyPoint > kp_inbound;
//	for(int i = 0; i < kp_vec_1.size(); i++) {
//		KeyPoint kp = kp_vec_1[i];
//		Mat hom_coord_vec = Mat::ones(3, 1, CV_32FC1);
//		hom_coord_vec.at<float>(0) = kp.pt.x;
//		hom_coord_vec.at<float>(1) = kp.pt.y;
//		Mat proj_kp_1_2 =  homography * hom_coord_vec;
//		proj_kp_1_2 /= proj_kp_1_2.at<float>(2);
//
//		// check within bounds
//		if(proj_kp_1_2.at<float>(0) >= 0 && proj_kp_1_2.at<float>(0) < img_2.cols && proj_kp_1_2.at<float>(1) >= 0
//				&& proj_kp_1_2.at<float>(1) < img_2.rows) {
//			kp_inbound.push_back(kp);
//		}
//	}
//
//	if(verbose) {
//		cout << "Found " << kp_inbound.size() << " keypoints for image 1 after removing ones that projected outside.\n";
//	}

	float min_correct_dist = -1, ave_correct_dist = 0, max_correct_dist = -1;
	float min_incorrect_dist = -1, ave_incorrect_dist = 0, max_incorrect_dist = -1;
	// Use ground truth homography to check whether descriptors are actually matching, using associated keypoints
	vector<DMatch> correct_matches;
	for(int i = 0; i < good_matches.size(); i++) {
		int kp_id_1 = good_matches[i].queryIdx;
		int kp_id_2 = good_matches[i].trainIdx;
		float dist = good_matches[i].distance;
		if(is_overlapping(kp_vec_1[kp_id_1], kp_vec_2[kp_id_2], homography, kp_dist_thresh)) {
			correct_matches.push_back(good_matches[i]);
			ave_correct_dist += dist;
			if(min_correct_dist == -1 || min_correct_dist > dist) {
				min_correct_dist = dist;
			}
			if(max_correct_dist == -1 || max_correct_dist < dist) {
				max_correct_dist = dist;
			}
#ifdef DEBUG
			is_good_match[good_matches[i].queryIdx] = "correct";
#endif
		}
		else {
			ave_incorrect_dist += dist;
			if(min_incorrect_dist == -1 || min_incorrect_dist > dist) {
				min_incorrect_dist = dist;
			}
			if(max_incorrect_dist == -1 || max_incorrect_dist < dist) {
				max_incorrect_dist = dist;
			}
		}
	}
	ave_correct_dist /= correct_matches.size();
	ave_incorrect_dist /= (good_matches.size() - correct_matches.size());

	if(verbose) {
		cout << correct_matches.size() << " correct matches found.\n";
	}

#ifdef DEBUG
	vector<float> correct_distances;
	vector<int> correct_distance_indices;
	bool correspondence_found;
#endif

	// Count total number of correspondences
	int num_correspondences = 0;
	for(int i = 0; i < kp_vec_1.size(); i++) {
#ifdef DEBUG
		correspondence_found = false;
#endif
		for(int j = 0; j < kp_vec_2.size(); j++) {
			if(is_overlapping(kp_vec_1[i], kp_vec_2[j], homography, kp_dist_thresh)) {
#ifdef DEBUG
				correct_distance_indices.push_back(j);
				correct_distances.push_back((float)norm(desc_1.row(i), desc_2.row(j)));
				correspondence_found = true;
#endif
				num_correspondences++;
				break;
			}
		}
#ifdef DEBUG
		if(!correspondence_found) {
			correct_distances.push_back(numeric_limits<double>::quiet_NaN());
			correct_distance_indices.push_back(-1);
		}
#endif
	}

	if(verbose) {
		cout << num_correspondences << " correct correspondences among the given keypoints.\n\n";
	}

#ifdef DEBUG
	ofstream debug_file;
	debug_file.open(results + "debug_" + img_1_name + "_to_" + img_2_name + ".csv");
	for(int i = 0; i < kp_vec_1.size(); i++) {
		debug_file << match_indices[i]            << ","
				   << match_distances[i]          << ","
				   << is_good_match[i]            << ","
				   << correct_distance_indices[i] << ","
				   << correct_distances[i]        << ",";
		if((is_good_match[i] == "correct" && correct_distance_indices[i] != -1)
				|| (is_good_match[i] == "no match" && correct_distance_indices[i] == -1)) {
			debug_file << "correct" << "\n";
		}
		else if((is_good_match[i] == "good" && correct_distance_indices[i] == -1)
				|| (is_good_match[i] == "no match" && correct_distance_indices[i] != -1)) {
			debug_file << "incorrect" << "\n";
		}
		else if((is_good_match[i] == "good" && correct_distance_indices[i] != -1)) {
			debug_file << "wrong keypoint" << "\n";
		}
	}
	debug_file.close();
#endif

	// Use data from last step to build metrics
	float match_ratio = (float)good_matches.size() / kp_vec_1.size();
	float precision = (float)correct_matches.size() / good_matches.size();
	float matching_score = match_ratio * precision;
	float recall = (float)correct_matches.size() / num_correspondences;
	int descriptor_size = desc_1.cols * desc_1.elemSize1();

	// Export metrics
	if(results == "") { // no directory specified for exporting results; print to console
		cout << desc_name << " descriptor results:\n";
		cout << "-----------------------\n";
		cout << "Match ratio: " << match_ratio << "\n";
		cout << "Matching score: " << matching_score << "\n";
		cout << "Precision: " << precision << "\n";
		cout << "Recall: " << recall << "\n";
	}
	else {
		ofstream f;
		f.open(results + "eval_" + img_1_name + "_to_" + img_2_name + ".txt");
		f << "Descriptor:                          " << desc_name              << "\n";
		f << "Image 1:                             " << argv[3]                << "\n";
		f << "Image 2:                             " << argv[6]                << "\n";
		f << "Descriptor Size (bytes):             " << descriptor_size        << "\n";
		f                                                                      << "\n";
		f << "Number of Keypoints for Image 1:     " << kp_vec_1.size()        << "\n";
//		f << "Number of Valid Projected Keypoints: " << kp_inbound.size()      << "\n";
		f << "Number of Keypoints for Image 2:     " << kp_vec_2.size()        << "\n";
		f << "Number of Matches:                   " << good_matches.size()    << "\n";
		f << "Number of Correct Matches:           " << correct_matches.size() << "\n";
		f << "Number of Correspondences:           " << num_correspondences    << "\n";
		f                                                                      << "\n";
		f << "Minimum Correct Distance:            " << min_correct_dist       << "\n";
		f << "Average Correct Distance:            " << ave_correct_dist       << "\n";
		f << "Maximum Correct Distance:            " << max_correct_dist       << "\n";
		f << "Minimum Incorrect Distance:          " << min_incorrect_dist     << "\n";
		f << "Average Incorrect Distance:          " << ave_incorrect_dist     << "\n";
		f << "Maximum Incorrect Distance:          " << max_incorrect_dist     << "\n";
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
	if(results != "") {
		Mat res;
		drawMatches(img_1, kp_vec_1, img_2, kp_vec_2, display_matches, res, Scalar::all(-1), Scalar::all(-1),
				vector< char >(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
		imwrite(results + "disp_" + img_1_name + "_to_" + img_2_name + ".png", res);
	}

	return 0;
}

// TODO: make a better type-adapatable version
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
