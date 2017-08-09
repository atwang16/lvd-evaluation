/*
 * generate_descriptors.cpp
 *
 *  Created on: Aug 3, 2017
 *      Author: Austin Wang
 *      Project: A Comparative Study of Local Visual Descriptors
 */

#include "detectors.hpp"
#include "descriptors.hpp"
#include "utils.hpp"
#include <opencv2/opencv.hpp>
#include <boost/filesystem.hpp>
#include <boost/algorithm/string.hpp>
#include <string>
#include <algorithm>
#include <set>

using namespace boost::filesystem;
using namespace std::chrono;
using namespace std;
using namespace cv;

#define PATH_DELIMITER "/"
#define BOOST_PATH_DELIMITER boost::is_any_of(PATH_DELIMITER)
#define IMG_READ_COLOR cv::IMREAD_GRAYSCALE

void compute(Descriptor d, string parameter_file, cv::Mat image, vector<KeyPoint>& keypoints, vector<Mat>& affine, cv::Mat& descriptors, double& desc_time);
bool is_image(path fname);
bool is_keypoint_file(path file_path);

int main(int argc, char *argv[]) {
	string parameter_file, image_directory, keypoint_directory, output_directory, database_name;
	bool single_image = false, overwrite = false;
	Descriptor d;
	map<string, Descriptor> detmap = {DESCRIPTOR_MAP};

	if(argc < 7) {
		cout << "Usage ./generate_descriptors descriptor path_to_parameter_file image_dataset_root_folder keypoint_dataset_root_folder path_to_destination overwrite_flag" << "\n";
		cout << "      ./generate_descriptors descriptor path_to_parameter_file path_to_image path_to_keypoint path_to_destination overwrite_flag" << "\n";
		return 1;
	}

	if(detmap.count(argv[1])) {
		d = detmap[argv[1]];
	}
	else {
		cout << "Error: descriptor not available. Aborting...\n";
		return 1;
	}
	parameter_file = argv[2];
	image_directory = argv[3];
	keypoint_directory = argv[4];
	output_directory = argv[5];
	overwrite = stoi(argv[6]);

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
				// Create folder in output directory for sequence
				path dir(output_directory + PATH_DELIMITER
						+ database_name + PATH_DELIMITER
						+ seq_name);
				boost::filesystem::create_directories(dir);

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

	double desc_time = 0;
	int num_images = 0, num_keypoints = 0;

	int i = 0;

	for(string img_p : image_paths) {
		cout << "Extracting descriptors for " << img_p << "\n";
		string kp_p, kp_name;
		string img_name = path(img_p).stem().string();
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

		string ds_path;
		if(single_image) {
			ds_path = output_directory + PATH_DELIMITER
					+ img_name + "_ds.csv";
		}
		else {
			// get the type name (e.g. ref, e1, e2, etc.)
			vector<string> img_split;
			boost::split(img_split, img_p, BOOST_PATH_DELIMITER);
			string seq_name = img_split[img_split.size() - 2];

			ds_path = output_directory + PATH_DELIMITER
					+ database_name + PATH_DELIMITER
					+ seq_name + PATH_DELIMITER
					+ img_name + "_ds.csv";
		}

		if(overwrite || !exists(ds_path)) {
			// Read the image and keypoint
			cv::Mat img_mat = cv::imread(img_p, IMG_READ_COLOR);
//			cout << kp_p << "\n";
			cv::Mat kp_mat_unproc = parse_file(kp_p, ',', CV_32F);
			kp_mat_unproc.convertTo(kp_mat_unproc, CV_32F);
			vector<cv::KeyPoint> keypoints;
			vector<cv::Mat> affine;

//			cout << kp_mat_unproc.at<float>(0, 0) << "\n";
//			for(int i = 0; i < kp_mat_unproc.rows; i++) {
//				for(int j = 0; j < 6; j++) {
//					cout << kp_mat_unproc.at<float>(i, j) << ",";
//				}
//				cout << "\n";
//			}

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

			// Compute descriptors
			cv::Mat descriptors;
			compute(d, parameter_file, img_mat, keypoints, affine, descriptors, desc_time);

			// Open the descriptor file for saving
			std::ofstream f_desc;
			f_desc.open(ds_path);
			f_desc << cv::format(descriptors, cv::Formatter::FMT_CSV) << endl;
			f_desc.close();
			cout << "Descriptors stored at " << ds_path << "\n" << "\n";

			num_images++;
			num_keypoints += keypoints.size();
		}
		else {
			cout << "Descriptor already exists at " << ds_path << "\n" << "\n";
		}
	}

	std::ofstream file;
	file.open(output_directory + PATH_DELIMITER
			+ database_name + PATH_DELIMITER
			+ "time.csv", std::ofstream::out | std::ofstream::app);
	file << "Descriptor generation per keypoint (mus)," << desc_time << "," << num_images << "," << num_keypoints << "\n";
	file.close();

	return 0;
}

void compute(Descriptor d, string parameter_file, cv::Mat image, vector<KeyPoint>& keypoints, vector<Mat>& affine, cv::Mat& descriptors, double& desc_time) {
	// Compute descriptors
	high_resolution_clock::time_point start = high_resolution_clock::now();
	(*d)(image, keypoints, affine, descriptors, parameter_file);
	high_resolution_clock::time_point desc_done = high_resolution_clock::now();

	int num_keypoints = keypoints.size();
	desc_time += (double)duration_cast<microseconds>(desc_done - start).count() / num_keypoints;
}

bool is_image(path fname) {
	string ext = extension(fname);
	return(ext == ".png" || ext == ".jpg" || ext == ".ppm" || ext == ".pgm");
}

bool is_keypoint_file(path file_path) {
	string stem = file_path.filename().string();
	return(stem.find("_kp.csv") != string::npos);
}
