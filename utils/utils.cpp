/*
 * utils.cpp
 *
 *  Created on: Jul 12, 2017
 *      Author: austin
 */

#include "utils.hpp"

using namespace std;
using namespace cv;

Mat parse_file(string fname, char delimiter, int type) {
	ifstream inputfile(fname);
	string current_line;

	vector< vector<Data> > all_data;

	// read each line
	while(getline(inputfile, current_line)) {
		if(current_line != "") {
			vector<Data> values;
			stringstream str_stream(current_line);
			string single_value;

			// Read each value with delimiter
			while(getline(str_stream, single_value, delimiter)) {
				if(single_value != "") {
					Data d;
					if(type == CV_32F || single_value.find(".") != string::npos || single_value.find("-") != string::npos) {
						type = CV_32F;
						d.f = stof(single_value);
					}
					else {
						d.u = stoi(single_value);
					}
					values.push_back(d);
				}
			}
			all_data.push_back(values);
		}
	}

	if(all_data.size()) {
		// Place data in OpenCV matrix
		Mat vect = Mat::zeros((int)all_data.size(), (int)all_data[0].size(), type);
		for(int row = 0; row < vect.rows; row++) {
		   for(int col = 0; col < vect.cols; col++) {
			   if(type == CV_32F) {
				   vect.at<float>(row, col) = all_data[row][col].f;
			   }
			   else {
				   vect.at<uint8_t>(row, col) = all_data[row][col].u;
			   }
		   }
		}
		return vect;
	}
	else {
		return cv::Mat();
	}
}

Mat parse_file(string fname, char delimiter) {
	return parse_file(fname, delimiter, CV_8U);
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

void load_parameters(string parameter_file, map<string, double>& params) {
	if(parameter_file != "null") {
		std::ifstream parameter_filestream(parameter_file);
		string line, value, var;
		std::vector<std::string> line_split;

		while(getline(parameter_filestream, line)) {
			boost::split(line_split, line, boost::is_any_of("="));
			var = line_split[0];
			std::transform(var.begin(), var.end(), var.begin(), ::tolower);
			value = line_split[1];

			if(params.count(var)) {
				params[var] = stod(value);
			}
		}
	}
}

bool dtob(double d) {
	return d < 0.5;
}
