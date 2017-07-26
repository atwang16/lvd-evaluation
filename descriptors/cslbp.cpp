/*
 * cslbp.cpp
 *
 *  Created on: Jul 25, 2017
 *      Author: Austin Wang
 *      Project: A Comparative Study of Local Visual Descriptors
 *      Descriptor: CS-LBP
 *      Descriptor Citation:  TBD
 *
 *  This source code extracts the CS-LBP descriptor from images.
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

#define PATH_DELIMITER "/"
#define BOOST_PATH_DELIMITER boost::is_any_of(PATH_DELIMITER)

/******************************
 * MODIFY FOR EACH DESCRIPTOR *
 ******************************/

// Compile-time Constants
#define IMG_READ_COLOR cv::IMREAD_GRAYSCALE

void compute(cv::Mat image, vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors, float const threshold) {
	descriptors = cv::Mat::zeros(keypoints.size(), 16, CV_8U);
	Mat image_fl;
	image.convertTo(image_fl, CV_32F);

	for(int i = 0; i < keypoints.size(); i++) {
		cv::Mat patch, patch_fl;
		cv::getRectSubPix(image_fl, cv::Size(keypoints[i].size, keypoints[i].size), keypoints[i].pt, patch);
		patch_fl = patch;
		for(int x = 1; x < patch.cols - 1; x++) {
			for(int y = 1; y < patch.rows - 1; y++) {
				int a = ((patch_fl.at<float>(x,y+1)   - patch_fl.at<float>(x,y-1)   > threshold) * 1);
				int b = ((patch_fl.at<float>(x+1,y+1) - patch_fl.at<float>(x-1,y-1) > threshold) * 2);
				int c = ((patch_fl.at<float>(x+1,y)   - patch_fl.at<float>(x-1,y)   > threshold) * 4);
				int d = ((patch_fl.at<float>(x+1,y-1) - patch_fl.at<float>(x-1,y+1) > threshold) * 8);
				descriptors.at<uchar>(i, a+b+c+d)++;
			}
		}
	}
}


void detectAndCompute(string descriptor, string parameter_file, cv::Mat image, vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors,
		long* kp_time, long* desc_time, long* total_time) {
	// Parameters to load from file
	std::ifstream params(parameter_file);
	std::string line, var, value;
	std::vector<std::string> line_split;

	// parameters with default values; modify for each descriptor
	float threshold = 0.1;

	// Load parameters from file
	while(getline(params, line)) {
		boost::split(line_split, line, boost::is_any_of("="));
		var = line_split[0];
		value = line_split.back();

		if(var == "THRESHOLD") {
			threshold = stoi(value);
		}
	}

	// Extract keypoints and compute descriptors
//	cv::Ptr< cv::FastFeatureDetector > fast = cv::FastFeatureDetector::create();
	high_resolution_clock::time_point start = high_resolution_clock::now();
	HesAffFeatureDetector(image, keypoints);
//	fast->detect(image, keypoints);
	high_resolution_clock::time_point kp_done = high_resolution_clock::now();
	cout << "Found " << keypoints.size() << " keypoints.\n";
	compute(image, keypoints, descriptors, threshold);
	high_resolution_clock::time_point desc_done = high_resolution_clock::now();
	cout << "Computed descriptors.\n";

	int num_keypoints = keypoints.size();
	if(num_keypoints > 0) {
		*kp_time += duration_cast<microseconds>(kp_done - start).count() / num_keypoints;
		*desc_time += duration_cast<microseconds>(desc_done - kp_done).count() / num_keypoints;
	}
	*total_time += duration_cast<milliseconds>(desc_done - start).count();
}

/***********************
 * DO NOT MODIFY BELOW *
 ***********************/

bool is_image(string fname);

int main(int argc, char *argv[]) {
	string parameter_file, image_directory, output_directory, database_name;
	bool single_image;

	// Get descriptor name
	string filename = __FILE__;
	vector<string> filename_split;
	boost::split(filename_split, filename, boost::is_any_of("/."));
	string descriptor = filename_split[filename_split.size() - 2];

	if(argc < 4) {
		cout << "Usage ./" << descriptor << " path_to_parameter_file image_dataset_root_folder path_to_destination" << "\n";
		cout << "      ./" << descriptor << " path_to_parameter_file path_to_image path_to_destination" << "\n";
		return 1;
	}

	parameter_file = argv[1];
	image_directory = argv[2];
	output_directory = argv[3];

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
						+ descriptor + PATH_DELIMITER
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
		cout << "Usage ./" << descriptor << " image_dataset_root_folder [destination_folder]" << "\n";
		cout << "      ./" << descriptor << " image_path [destination_folder]" << "\n";
		return 1;
	}

	long kp_time = 0, desc_time = 0, total_time = 0;
	int num_images = 0;

	for(auto const& img : images) {
		cout << "Extracting descriptors for " << img << "\n";

		// Read the image
		cv::Mat img_mat = cv::imread(img, IMG_READ_COLOR);

		string ds_path, kp_path;
		vector<string> img_split;
		boost::split(img_split, img, boost::is_any_of("/."));
		string img_name = img_split[img_split.size() - 2];

		if(single_image) {
			ds_path = output_directory + PATH_DELIMITER
					+ img_name + "_ds.csv";
			kp_path = output_directory + PATH_DELIMITER
					+ img_name + "_kp.csv";
		}
		else {
			// get the type name (e.g. ref, e1, e2, etc.)
			string seq_name = img_split[img_split.size() - 3];

			ds_path = output_directory + PATH_DELIMITER
					+ descriptor + PATH_DELIMITER
					+ database_name + PATH_DELIMITER
					+ seq_name + PATH_DELIMITER
					+ img_name + "_ds.csv";
			kp_path = output_directory + PATH_DELIMITER
					+ descriptor + PATH_DELIMITER
					+ database_name + PATH_DELIMITER
					+ seq_name + PATH_DELIMITER
					+ img_name + "_kp.csv";
		}

		// Compute descriptors
		cv::Mat descriptors;
		vector<cv::KeyPoint> keypoints;
		detectAndCompute(descriptor, parameter_file, img_mat, keypoints, descriptors, &kp_time, &desc_time, &total_time);

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

		// Open the descriptor file for saving
		std::ofstream f_desc;
		f_desc.open(ds_path);
		f_desc << cv::format(descriptors, cv::Formatter::FMT_CSV) << endl;
		f_desc.close();
		cout << "Descriptors stored at " << ds_path << "\n" << "\n";

		num_images++;
	}

	kp_time /= num_images;
	desc_time /= num_images;
	total_time /= num_images;

	std::ofstream f;
	f.open(output_directory + PATH_DELIMITER
			+ descriptor + PATH_DELIMITER
			+ database_name + PATH_DELIMITER
			+ "time.txt");
	f << "Average time to detect keypoints, per keypoint:    " << kp_time << " microseconds" << "\n";
	f << "Average time to extract descriptors, per keypoint: " << desc_time << " microseconds" << "\n";
	f << "Average time per image:                            " << total_time << " milliseconds" << "\n";
	f << "Number of images:                                  " << num_images << "\n";
	f.close();

	return 0;
}

bool is_image(string fname) {
	string ext = extension(fname);
	return(ext == ".png" || ext == ".jpg" || ext == ".ppm" || ext == ".pgm");
}