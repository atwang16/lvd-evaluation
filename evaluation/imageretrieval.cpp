/*
 * basetest.cpp
 *
 *  Created on: Jun 27, 2017
 *      Author: austin
 */

#include <opencv2/opencv.hpp>
#include <cstdint>
#include <cmath>
#include <string>
#include <iostream>
#include <sstream>
#include <fstream>
#include <cmath>
#include <boost/filesystem.hpp>
#include <boost/algorithm/string.hpp>
#include <chrono>

using namespace std;
using namespace cv;
using namespace std::chrono;
using namespace boost::filesystem;

Mat parse_file(string fname, char delimiter, int type);
int get_dist_metric(string metric);
bool is_overlapping(KeyPoint kp_1, KeyPoint kp_2, Mat hom, float threshold);
bool is_desc(std::string fname);
bool is_hidden_file(std::string fname);
string get_cat(string desc_fname);

int main(int argc, char *argv[]) {
	Mat query_desc, database_desc;
	string desc_name, db_path, query_cat;
	vector<KeyPoint> kp_vec_1, kp_vec_2;
	float dist_ratio_thresh = 0.8f, fixed_dist_thresh = INFINITY;
	int dist_metric, verbose_output = 0;

	if(argc < 8) {
		cout << "Usage ./imageretrieval parameters_file desc_name dist_threshold dist_metric query_descs desc_database verbose_output\n";
		return -1;
	}

	// Load parameters for base test
	std::ifstream params(argv[1]);
	string line, var, value;
	std::vector<std::string> line_split;

	while(getline(params, line)) {
		boost::split(line_split, line, boost::is_any_of("="));
		var = line_split[0];
		value = line_split.back();
		if(var == "DIST_RATIO_THRESH") {
			dist_ratio_thresh = stof(value);
		}
//		else if(var == "VERBOSE") {
//			verbose = stof(value);
//		}
	}

	// Parse remaining arguments
	desc_name = argv[2];
	fixed_dist_thresh = stof(argv[3]);
	dist_metric = get_dist_metric(argv[4]);

	query_desc = parse_file(argv[5], ',', CV_8U);
	query_cat = get_cat(argv[5]);

	// Extract all descriptors from database
	vector<string> desc_vec;
	db_path = argv[6];

	verbose_output = atoi(argv[7]);

	if(is_directory(db_path)) {
		vector<string> db_subdirs;

		// iterate through sequence folders of directory
		for(auto& entry : boost::make_iterator_range(directory_iterator(db_path), {})) {
			db_subdirs.push_back(entry.path().string()); // appends path of file to seqs vector
		}

		for(auto const& cat: db_subdirs) {
			vector<string> cat_split;
//			boost::split(cat_split, cat, boost::is_any_of("/"));
//			string seq_name = cat_split.back();
			if(is_directory(cat)) {
				for(auto& entry : boost::make_iterator_range(directory_iterator(cat), {})) {
					string fname = entry.path().string();
					if(is_desc(fname)) {
						desc_vec.push_back(fname);
					}
				}
			}
		}
	}

	// Calculate number of correspondences for each database image
	vector<int> num_corr;
	int nc = 0;
	vector< vector<DMatch> > nn_matches;
	Ptr< BFMatcher > matcher = BFMatcher::create(dist_metric); // no cross check

	for(int i = 0; i < desc_vec.size(); i++) {
		Mat database_desc = parse_file(desc_vec[i], ',', CV_8U);
		matcher->knnMatch(query_desc, database_desc, nn_matches, 2);

		nc = 0;
		for(int i = 0; i < nn_matches.size(); i++) {
			if(nn_matches[i][0].distance < fixed_dist_thresh && nn_matches[i][0].distance < dist_ratio_thresh * nn_matches[i][1].distance) {
				nc++;
			}
		}
		num_corr.push_back(nc);
		vector< vector<DMatch> >().swap(nn_matches);
	}

	// Find permutation of number of correspondences
	vector<int> ind;
	for(int i = 0; i < desc_vec.size(); i++) {
		ind.push_back(i);
	}
	sort(ind.begin(), ind.end(),
			[&](const int& i, const int& j) {
        		return (num_corr[i] > num_corr[j]);
    		}
	);

	// Create vector of rank labels to account for ties
	vector<int> rank;
	rank.push_back(1); // initial value
	for(int ord = 2; ord <= ind.size(); ord++) {
		if(num_corr[ind[ord - 1]] == num_corr[ind[ord - 2]]) { // if equal number of matches is found
			rank.push_back(rank[rank.size() - 1]);
		}
		else {
			rank.push_back(ord);
		}
	}

	// Use data from last step to build metrics
	float ave_precision = 0;
	int num_rel_imgs = 0;
	bool is_first_img_rel = false;
	for(int ord = 1; ord <= ind.size(); ord++) {
		int i = ind[ord - 1];
		if(query_cat == get_cat(desc_vec[i])) {
			num_rel_imgs++;
			ave_precision += (float)num_rel_imgs / rank[ord - 1];
			if(ord == 1) {
				is_first_img_rel = true;
			}
		}
	}
	ave_precision /= num_rel_imgs;

	// Export metrics
	if(verbose_output) {
		cout << desc_name << " descriptor results:\n";
		cout << "-----------------------"                                  << "\n";
		cout << "Average precision: " << ave_precision                     << "\n";
		cout << "Success:           " << (is_first_img_rel ? "yes" : "no") << "\n";
		cout                                                               << "\n";
		for(int ord = 1; ord <= ind.size(); ord++) {
			int i = ind[ord - 1];
			cout << setw(5) << right << rank[ord - 1] << " ";
			cout << setw(5) << right << num_corr[i] << " ";
			cout << desc_vec[i] << "\n";
		}
	}
	else {
		cout << ave_precision << " " << (is_first_img_rel ? 1 : 0);
	}

	return 0;
}

// TODO: make a better type-adapatable version
Mat parse_file(string fname, char delimiter, int type) {
	std::ifstream inputfile(fname);
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

bool is_hidden_file(std::string fname) {
    return(fname[0] == '.');
}

bool is_desc(std::string fname) {
	vector<string> fname_split;
	boost::split(fname_split, fname, boost::is_any_of("_"));
	return(fname_split.back() == "descriptor.csv");
}

string get_cat(string desc_fname) {
	vector<string> path_split;
	boost::split(path_split, desc_fname, boost::is_any_of("/"));
	return(path_split[path_split.size() - 2]);
}
