/*
 * generate_keypoints.cpp
 *
 *  Created on: Jun 26, 2017
 *      Author: Austin Wang
 *      Project: A Comparative Study of Local Visual Descriptors
 *      Descriptor: SIFT
 *      Descriptor Citation:  David G. Lowe. Object recognition from local scale-invariant features. In Proceedings
 *                            of 1999 IEEE International Conference on Computer Vision, pages 1150–1157, September 1999.
 *
 *  This source code generates keypoints using a chosen keypoint detector and outputs the data to a file.
 */

#include <opencv2/opencv.hpp>
#include <string>
#include <iostream>
#include <sstream>
#include <fstream>
#include <boost/filesystem.hpp>
#include <boost/algorithm/string.hpp>
#include <chrono>
#include <algorithm>
#include "detectors.hpp"

using namespace boost::filesystem;
using namespace std::chrono;
using namespace std;
using namespace cv;

#define PATH_DELIMITER "/"
#define BOOST_PATH_DELIMITER boost::is_any_of(PATH_DELIMITER)

/******************************
 * MODIFY FOR EACH DESCRIPTOR *
 ******************************/

// Libraries
#include "opencv2/xfeatures2d.hpp"

// Compile-time Constants
#define IMG_READ_COLOR cv::IMREAD_GRAYSCALE

void detect(string parameter_file, cv::Mat image, vector<KeyPoint>& keypoints, long& kp_time) {
	// Parameters to load from file
	std::ifstream params(parameter_file);
	std::string line, var, value;
	std::vector<std::string> line_split;

	// parameters with default values
	int nfeatures = 0; // 0 for SIFT implies no upper bound
	int nOctaveLayers = 3;
	double contrastThreshold = 0.04;
	double edgeThreshold = 10;
	double sigma = 1.6;

	// Load parameters from file
	while(getline(params, line)) {
		boost::split(line_split, line, boost::is_any_of("="));
		var = line_split[0];
		value = line_split[1];

		if(var == "N_FEATURES") {
			nfeatures = stoi(value);
		}
		else if(var == "N_OCTAVE_LAYERS") {
			nOctaveLayers = stoi(value);
		}
		else if(var == "CONTRAST_THRESHOLD") {
			contrastThreshold = stod(value);
		}
		else if(var == "EDGE_THRESHOLD") {
			edgeThreshold = stod(value);
		}
		else if(var == "SIGMA") {
			sigma = stod(value);
		}
	}

	// Extract keypoints and compute descriptors
	cv::Ptr<cv::Feature2D> sift = cv::xfeatures2d::SIFT::create(nfeatures, nOctaveLayers, contrastThreshold, edgeThreshold, sigma);
	high_resolution_clock::time_point start = high_resolution_clock::now();
	sift->detectAndCompute(image, cv::noArray(), keypoints, cv::noArray(), false);
	high_resolution_clock::time_point kp_done = high_resolution_clock::now();

	int num_keypoints = keypoints.size();
	if(num_keypoints > 0) {
		kp_time += duration_cast<microseconds>(kp_done - start).count() / num_keypoints;
	}
}

/***********************
 * DO NOT MODIFY BELOW *
 ***********************/

bool is_image(string fname);

int main(int argc, char *argv[]) {
	string parameter_file, image_directory, output_directory, database_name;
	bool single_image;
	Detector det;

	if(argc < 4) {
		cout << "Usage ./generate_keypoints detector path_to_parameter_file image_dataset_root_folder path_to_destination" << "\n";
		cout << "      ./generate_keypoints detector path_to_parameter_file path_to_image path_to_destination" << "\n";
		return 1;
	}

	det = stoi(argv[1]);
	parameter_file = argv[2];
	image_directory = argv[3];
	output_directory = argv[4];

	// get all the sequences
	vector<string> images;

	if(is_directory(image_directory)) {
		single_image = false;
		cout << "Database: " << image_directory << "\n";
		vector<string> img_dir_split;
		boost::split(img_dir_split, image_directory, BOOST_PATH_DELIMITER);
		database_name = img_dir_split.back();

		// iterate through sequence folders of directory
		vector<string> sequences;
		for(auto& entry : boost::make_iterator_range(directory_iterator(image_directory), {})) {
			sequences.push_back(entry.path().string()); // appends path of file to seqs vector
		}

		for(auto const& seq: sequences) {
			vector<string> seq_split;
			boost::split(seq_split, seq, BOOST_PATH_DELIMITER);
			string seq_name = seq_split.back();
			if(is_directory(seq)) {
				boost::filesystem::path dir(output_directory + PATH_DELIMITER
						+ database_name + PATH_DELIMITER
						+ seq_name);
				boost::filesystem::create_directories(dir);

				for(auto& entry : boost::make_iterator_range(directory_iterator(seq), {})) {
					string fname = entry.path().string();
					if(is_image(fname)) {
						images.push_back(fname);
					}
				}
			}
		}
		sort(images.begin(), images.end());
	}
	else if(is_image(image_directory)){ // input is an image
		single_image = true;
		images.push_back(image_directory);
	}
	else {
		cout << "Error: invalid input.\n";
		cout << "Usage ./generate_keypoints detector path_to_parameter_file image_dataset_root_folder path_to_destination" << "\n";
		cout << "      ./generate_keypoints detector path_to_parameter_file path_to_image path_to_destination" << "\n";
		return 1;
	}

	long kp_time = 0;
	int num_images = 0;

	for(auto const& img : images) {
		cout << "Extracting keypoints for " << img << "\n";

		// Read the image
		cv::Mat img_mat = cv::imread(img, IMG_READ_COLOR);

		string kp_path;
		vector<string> img_split;
		boost::split(img_split, img, boost::is_any_of("/."));
		string img_name = img_split[img_split.size() - 2];

		if(single_image) {
			kp_path = output_directory + PATH_DELIMITER
					+ img_name + "_kp.csv";
		}
		else {
			// get the type name (e.g. ref, e1, e2, etc.)
			string seq_name = img_split[img_split.size() - 3];

			kp_path = output_directory + PATH_DELIMITER
					+ database_name + PATH_DELIMITER
					+ seq_name + PATH_DELIMITER
					+ img_name + "_kp.csv";
		}

		// Compute descriptors
		vector<KeyPoint> keypoints;
		detect(parameter_file, img_mat, keypoints, kp_time);

		// Open the keypoint file for saving
		std::ofstream f_key;
		f_key.open(kp_path);
		for(int i = 0; i < keypoints.size(); i++) {
			f_key << keypoints[i].pt.x     << ",";
			f_key << keypoints[i].pt.y     << ",";
			f_key << keypoints[i].size     << ",";
			f_key << keypoints[i].angle    << ",";
			f_key << keypoints[i].response << ",";
			f_key << keypoints[i].octave   << endl;
		}
		f_key.close();
		cout << "Keypoints stored at " << kp_path << "\n";

		num_images++;
	}

	std::ofstream f;
	f.open(output_directory + PATH_DELIMITER
			+ database_name + PATH_DELIMITER
			+ "time.csv");
	f << kp_time << "," << (kp_time / num_images) << "\n"; // Average time to detect keypoints, per keypoint
	f.close();

	return 0;
}

bool is_image(string fname) {
	string ext = extension(fname);
	return(ext == ".png" || ext == ".jpg" || ext == ".ppm" || ext == ".pgm");
}
