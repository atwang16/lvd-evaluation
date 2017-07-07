/*
 * akaze.cpp
 *
 *  Created on: Jul 5, 2017
 *      Author: Austin Wang
 *      Project: A Comparative Study of Local Visual Descriptors
 *      Descriptor: A-KAZE
 *      Descriptor Citation:  <To Be Entered>
 *      Template for code adapted from
 *          Vassileios Balntas, Karel Lenc, Andrea Vedaldi, and Krystian Mikolajczyk. HPatch: A benchmark and evaluation
 *          of handcrafted and learned local descriptor. In Proceedings of 2017 IEEE Conference on Computer Vision and
 *          Pattern Recognition, July 2017.
 *
 *  This is the source code for an executable for extracting descriptors from whole images. To run the executable,
 *  pass a single argument to the executable containing a path to the file directory with all of the images.
 */

#include <opencv2/opencv.hpp>
#include <string>
#include <iostream>
#include <sstream>
#include <fstream>
#include <boost/filesystem.hpp>
#include <boost/algorithm/string.hpp>
#include <chrono>

using namespace boost::filesystem;
using namespace std::chrono;

#define DEFAULT_OUTPUT_DIR "results/"

/******************************
 * MODIFY FOR EACH DESCRIPTOR *
 ******************************/

// Compile-time Constants
#define DESCRIPTOR "akaze"    // name of descriptor
#define IMG_READ_COLOR cv::IMREAD_GRAYSCALE
#define PATH_TO_DESCRIPTORS_FOLDER "/Users/austin/MIT/02_Spring_2017/MISTI_France/lvd-evaluation/descriptors/"

void detectAndCompute(cv::Mat image, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors, long* kp_time, long* desc_time, long* total_time) {
	// Parameters to load from file
	ifstream params(PATH_TO_DESCRIPTORS_FOLDER DESCRIPTOR "_parameters.txt");
	std::string line, var, value;
	std::vector<std::string> line_split;

	// default parameters
	int descriptor_type=cv::AKAZE::DESCRIPTOR_MLDB;
	int descriptor_size=0;
	int descriptor_channels=3;
	float threshold=0.001f;
	int nOctaves=4;
	int nOctaveLayers=4;
	int diffusivity=cv::KAZE::DIFF_PM_G2;

	// Load parameters from file
	while(getline(params, line)) {
		boost::split(line_split, line, boost::is_any_of("="));
		var = line_split[0];
		value = line_split.back();
		if(var == "descriptor_type") {
			descriptor_type = stoi(value);
		}
		else if(var == "descriptor_size") {
			descriptor_size = stoi(value);
		}
		else if(var == "descriptor_channels") {
			descriptor_channels = stoi(value);
		}
		else if(var == "threshold") {
			threshold = stof(value);
		}
		else if(var == "nOctaves") {
			nOctaves = stoi(value);
		}
		else if(var == "nOctaveLayers") {
			nOctaveLayers = stoi(value);
		}
		else if(var == "diffusivity") {
			diffusivity = stoi(value);
		}
	}

	// Initialization
	cv::Ptr<cv::Feature2D> akaze = cv::AKAZE::create(descriptor_type, descriptor_size, descriptor_channels, threshold, nOctaves, nOctaveLayers, diffusivity);

	high_resolution_clock::time_point start = high_resolution_clock::now();

	// detect keypoints
	akaze->detect(image, keypoints, cv::noArray());

	high_resolution_clock::time_point kp_done = high_resolution_clock::now();

	// extract descriptors
	akaze->detectAndCompute(image, cv::noArray(), keypoints, descriptors, false); // bug seems to make separate detection and extraction not work

	high_resolution_clock::time_point desc_done = high_resolution_clock::now();

	int num_keypoints = keypoints.size();
	*kp_time += duration_cast<microseconds>(kp_done - start).count() / num_keypoints;
	*desc_time += duration_cast<microseconds>(desc_done - kp_done).count() / num_keypoints;
	*total_time += duration_cast<milliseconds>(desc_done - start).count();
}

/***********************
 * DO NOT MODIFY BELOW *
 ***********************/

bool is_image(std::string fname);
bool is_hidden_file(std::string fname);

int main(int argc, char *argv[]) {
	std::string output_directory = DEFAULT_OUTPUT_DIR;
	std::string main_dir;

	if(argc < 2) {
		std::cout << "Usage ./" << DESCRIPTOR << " image_dataset_root_folder [destination_folder]" << "\n";
		std::cout << "      ./" << DESCRIPTOR << " image_path [destination_folder]" << "\n";
		return -1;
	}
	else if(argc >= 3) {
		output_directory = argv[2];
	}

	// get all the sequences
	std::string descr_name = DESCRIPTOR;
	std::string input = argv[1];
	std::vector<std::string> img_vec;

	if(is_directory(input)) {
		std::vector<std::string> seq_vec;
		std::cout << "Database: " << input << "\n";
		std::vector<std::string> input_split;
		boost::split(input_split, input, boost::is_any_of("/"));
		main_dir = input_split[input_split.size() - 2] + "/";

		// iterate through sequence folders of directory
		for(auto& entry : boost::make_iterator_range(directory_iterator(input), {})) {
			seq_vec.push_back(entry.path().string()); // appends path of file to seqs vector
		}

		for(auto const& seq: seq_vec) {
			std::vector<std::string> seq_split;
			boost::split(seq_split, seq, boost::is_any_of("/"));
			std::string seq_name = seq_split.back();
			if(!is_hidden_file(seq_name)) {
				boost::filesystem::path dir(output_directory + descr_name + "/" + main_dir + seq_name);
				boost::filesystem::create_directories(dir);

				for(auto& entry : boost::make_iterator_range(directory_iterator(seq), {})) {
					std::string fname = entry.path().string();
					if(is_image(fname)) {
						img_vec.push_back(fname);
					}
				}
			}
		}
	}
	else if(is_image(input)){ // input is an image
		img_vec.push_back(input);
		boost::filesystem::path dir(output_directory + descr_name + "/" + main_dir);
		boost::filesystem::create_directories(dir);
	}
	else {
		std::cout << "Error: invalid input.\n";
		std::cout << "Usage ./" << DESCRIPTOR << " image_dataset_root_folder [destination_folder]" << "\n";
		std::cout << "      ./" << DESCRIPTOR << " image_path [destination_folder]" << "\n";
		return -1;
	}

	long kp_time = 0, desc_time = 0, total_time = 0;
	int num_images = 0;

	for(auto const& fname : img_vec) {
		std::cout << "Extracting descriptors for " << fname << "\n";

		// Read the image
		cv::Mat img = cv::imread(fname, IMG_READ_COLOR);

		std::string dt_fname_desc, dt_fname_key;

		// get the type name (e.g. ref, e1, e2, etc.)
		std::vector<std::string> strs;
		boost::split(strs, fname, boost::is_any_of("/"));
		std::string img_name = strs.back();
		std::string seq_name = is_directory(input) ? strs[strs.size() - 2] : "";

		// parse out periods
		std::vector<std::string> strs_;
		boost::split(strs_, img_name, boost::is_any_of("."));
		std::string tp =  strs_[0];

		dt_fname_desc = output_directory + descr_name + "/" + main_dir + seq_name + "/" + tp + "_descriptor.csv";
		dt_fname_key = output_directory + descr_name + "/" + main_dir + seq_name + "/" + tp + "_keypoint.csv";

		// Compute descriptors
		cv::Mat descriptors;
		std::vector<cv::KeyPoint> keypoints;
		detectAndCompute(img, keypoints, descriptors, &kp_time, &desc_time, &total_time);

		// Open the keypoint file for saving
		std::ofstream f_key;
		f_key.open(dt_fname_key);
		for(int i = 0; i < keypoints.size(); i++) {
			f_key << keypoints[i].pt.x     << ",";
			f_key << keypoints[i].pt.y     << ",";
			f_key << keypoints[i].size     << ",";
			f_key << keypoints[i].angle    << ",";
			f_key << keypoints[i].response << ",";
			f_key << keypoints[i].octave   << std::endl;
		}
		f_key.close();
		std::cout << "Keypoints stored at " << dt_fname_key << "\n";

		// Open the descriptor file for saving
		std::ofstream f_desc;
		f_desc.open(dt_fname_desc);
		f_desc << cv::format(descriptors, cv::Formatter::FMT_CSV) << std::endl;
		f_desc.close();
		std::cout << "Descriptors stored at " << dt_fname_desc << "\n" << "\n";

		num_images++;
	}

	kp_time /= num_images;
	desc_time /= num_images;
	total_time /= num_images;

	ofstream f;
	f.open(output_directory + descr_name + "/" + main_dir + "time.txt");
	f << "Average time to detect keypoints, per keypoint:    " << kp_time << " microseconds" << "\n";
	f << "Average time to extract descriptors, per keypoint: " << desc_time << " microseconds" << "\n";
	f << "Average time per image:                            " << total_time << " milliseconds" << "\n";
	f.close();

	return 0;
}

bool is_image(std::string fname) {
	std::string ext = boost::filesystem::extension(fname);
	return(ext == ".png" || ext == ".jpg" || ext == ".ppm" || ext == ".pgm");
}

bool is_hidden_file(std::string fname) {
    return(fname[0] == '.');
}
