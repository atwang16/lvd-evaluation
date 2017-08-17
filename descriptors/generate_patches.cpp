/*
 * generate_patches.cpp
 *
 *  Created on: Aug 11, 2017
 *      Author: Austin Wang
 *      Project: A Comparative Study of Local Visual Descriptors
 */

#include "detectors.hpp"
#include "utils.hpp"
#include <opencv2/opencv.hpp>
#include <boost/filesystem.hpp>
#include <boost/algorithm/string.hpp>
#include <iostream>
#include <string>
#include <sstream>
#include <algorithm>
#include <set>
#include <mat.h>
#include <matrix.h>

using namespace boost::filesystem;
using namespace std;
using namespace cv;

#define PATH_DELIMITER "/"
#define BOOST_PATH_DELIMITER boost::is_any_of(PATH_DELIMITER)
#define IMG_READ_COLOR cv::IMREAD_GRAYSCALE
#define IMG_OUTPUT 0
#define MAT_OUTPUT 1

bool is_image(path fname);
bool is_keypoint_file(path file_path);

int main(int argc, char *argv[]) {
	string parameter_file, image_directory, keypoint_directory, output_directory, database_name;
	int patch_size = 0, type = IMG_OUTPUT;
	bool single_image = false;

	if(argc < 6) {
		cout << "Usage ./generate_patches image_dataset_root_folder keypoint_dataset_root_folder path_to_destination patch_size type" << "\n";
		cout << "      ./generate_patches path_to_image path_to_keypoint path_to_destination patch_size type" << "\n";
		return 1;
	}

	image_directory = argv[1];
	keypoint_directory = argv[2];
	output_directory = argv[3];
	patch_size = stoi(argv[4]);
	type = stoi(argv[5]);
	if(type != IMG_OUTPUT && type != MAT_OUTPUT) {
		cout << "Error: invalid type. Aborting...\n";
		return 1;
	}

	// get all the sequences
	vector<string> image_paths, keypoint_paths;

	if(is_directory(image_directory) && is_directory(keypoint_directory)) {
		database_name = path(image_directory).stem().string();

		// iterate through sequence folders of image directory
		vector<string> img_sequences;
		for(auto& entry : boost::make_iterator_range(directory_iterator(image_directory), {})) {
			img_sequences.push_back(entry.path().string()); // appends path of file to seqs vector
		}
		sort(img_sequences.begin(), img_sequences.end());

		// iterate through sequence folders of keypoint directory
		vector<path> kp_sequences;
		for(auto& entry : boost::make_iterator_range(directory_iterator(keypoint_directory), {})) {
			kp_sequences.push_back(entry.path()); // appends path of file to seqs vector
		}

		for(auto& seq : kp_sequences) {
			string seq_name = path(seq).stem().string();
			if(is_directory(seq)) {
				// Enumerate all of the keypoint files
				for(auto& entry : boost::make_iterator_range(directory_iterator(seq), {})) {
					path file_path = entry.path();
					if(is_keypoint_file(file_path)) {
						keypoint_paths.push_back(file_path.string());
					}
				}
			}
		}
		sort(keypoint_paths.begin(), keypoint_paths.end());

		for(auto const& seq: img_sequences) {
			string seq_name = path(seq).stem().string();
			if(is_directory(seq)) {
				// Enumerate all of the image files
				for(auto& entry : boost::make_iterator_range(directory_iterator(seq), {})) {
					path fname = entry.path();
					if(is_image(fname)) {
						image_paths.push_back(fname.string());
					}
				}
			}
		}
		sort(image_paths.begin(), image_paths.end());
	}
	else if(is_image(image_directory) && is_keypoint_file(keypoint_directory)){ // input is a single file
		single_image = true;
		image_paths.push_back(image_directory);
		keypoint_paths.push_back(keypoint_directory);
	}
	else {
		cout << "Error: invalid input; must be an image or image directory.\n";
		return 1;
	}

	int i = 0;
	string img_dir, patch_path;

	for(string img_p : image_paths) {
		cout << "Extracting patches for " << img_p << "\n";
		string kp_p, kp_name;
		string img_name = path(img_p).stem().string();
		string img_ext = path(img_p).extension().string();
		do {
			if(i < keypoint_paths.size()) {
				kp_p = keypoint_paths[i++];
				kp_name = path(kp_p).stem().string();
			}
			else {
				continue;
			}
		} while(kp_name.find(img_name) == string::npos);

		if(i > keypoint_paths.size()) {
			break;
		}

		img_dir = output_directory;
		if(single_image) {
			if(type == IMG_OUTPUT) {
				img_dir += PATH_DELIMITER + img_name; // create directory
				path dir(img_dir);
				boost::filesystem::create_directories(dir);
			}
		}
		else {
			// get sequence name
			vector<string> img_split;
			boost::split(img_split, img_p, BOOST_PATH_DELIMITER);
			string seq_name = img_split[img_split.size() - 2];

			img_dir = output_directory + PATH_DELIMITER
					+ database_name + PATH_DELIMITER
					+ seq_name;
			if(type == IMG_OUTPUT) {
				img_dir += PATH_DELIMITER + img_name; // create directory
				path dir(img_dir);
				boost::filesystem::create_directories(dir);
			}
		}

		// Read the image and keypoint
		cv::Mat img_mat = cv::imread(img_p, IMG_READ_COLOR);
		cv::Mat kp_mat_unproc = parse_file(kp_p, ',');
		kp_mat_unproc.convertTo(kp_mat_unproc, CV_32F);
		vector<cv::KeyPoint> keypoints;
		vector<cv::Mat> affine;

		KeyPointCollection kp_col;
		for(int i = 0; i < kp_mat_unproc.rows; i++) {
			cv::KeyPoint kp = cv::KeyPoint();
			cv::Mat aff = cv::Mat::zeros(2, 3, CV_32F);
			kp.pt.x = kp_mat_unproc.at<float>(i, 0);
			kp.pt.y = kp_mat_unproc.at<float>(i, 1);
			kp.size = kp_mat_unproc.at<float>(i, 2);
			kp.angle = kp_mat_unproc.at<float>(i, 3);
			kp.response = kp_mat_unproc.at<float>(i, 4);
			kp.octave = kp_mat_unproc.at<float>(i, 5);
			kp.class_id = kp_mat_unproc.at<float>(i, 6);
			aff.at<float>(0, 0) = kp_mat_unproc.at<float>(i, 7);
			aff.at<float>(0, 1) = kp_mat_unproc.at<float>(i, 8);
			aff.at<float>(0, 2) = kp_mat_unproc.at<float>(i, 9);
			aff.at<float>(1, 0) = kp_mat_unproc.at<float>(i, 10);
			aff.at<float>(1, 1) = kp_mat_unproc.at<float>(i, 11);
			aff.at<float>(1, 2) = kp_mat_unproc.at<float>(i, 12);
			keypoints.push_back(kp);
			affine.push_back(aff);
		}
		kp_col.keypoints = keypoints;
		kp_col.affine = affine;

		// Compute patches
		if(type == IMG_OUTPUT) {
			for(int k = 0; k < kp_col.keypoints.size(); k++) {
				std::stringstream ss;
				ss << std::setw(5) << std::setfill('0') << k;
				patch_path = img_dir + PATH_DELIMITER
						+ img_name + "_pa_" + ss.str() + img_ext;
				Mat patch = get_patch(img_mat, patch_size, kp_col.keypoints[k].pt, kp_col.affine[k]);
				imwrite(patch_path, patch);
			}
		}
		else {
			MATFile *patches;
			mxArray *arr;
			float *arr_ptr;
			patch_path = img_dir + PATH_DELIMITER
					   + img_name + "_pa.mat";
			const unsigned long int dimensions[4] = {kp_col.keypoints.size(), 1, (unsigned long int)patch_size, (unsigned long int)patch_size};
			if((patches = matOpen(patch_path.c_str(), "w")) == NULL) {
				cout << "Error creating file " << patch_path << "\n";
				return 1;
			}

			if((arr = mxCreateNumericArray(4, dimensions, mxDOUBLE_CLASS, mxREAL)) == NULL) {
				cout << "Error creating numeric array.\n";
				return 1;
			}
			arr_ptr = (float *)arr;

			for(int k = 0; k < kp_col.keypoints.size(); k++) {
				Mat p = get_patch(img_mat, patch_size, kp_col.keypoints[k].pt, kp_col.affine[k]);
				for(int c = 0; c < p.cols; c++) {
					Mat p_trans;
					transpose(p, p_trans);
					memcpy(arr_ptr, (void *)p_trans.ptr<float>(c), sizeof(CV_32F) * p.rows);
					arr_ptr += p.rows;

				}
			}

			if(matPutVariable(patches, "Patches", arr) != 0) {
				cout << "Error loading patches into file.\n";
				return 1;
			}

			if(matClose(patches) != 0) {
				cout << "Error closing file " << patch_path << "\n";
				return 1;
			}
		}
		cout << "Patches stored at " << patch_path << "\n\n";
	}

	return 0;
}

bool is_image(path fname) {
	string ext = extension(fname);
	return(ext == ".png" || ext == ".jpg" || ext == ".ppm" || ext == ".pgm");
}

bool is_keypoint_file(path file_path) {
	string stem = file_path.filename().string();
	return(stem.find("_kp.csv") != string::npos);
}
