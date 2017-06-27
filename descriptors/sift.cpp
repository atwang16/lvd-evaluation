/*
 * sift.cpp
 *
 *  Created on: Jun 26, 2017
 *      Author: Austin Wang
 *      Project: A Comparative Study of Local Visual Descriptors
 *      Descriptor: SIFT
 *      Descriptor Citation:  David G. Lowe. Object recognition from local scale-invariant features. In Proceedings
 *                            of 1999 IEEE International Conference on Computer Vision, pages 1150–1157, September 1999.
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

using namespace boost::filesystem;

/******************************
 * MODIFY FOR EACH DESCRIPTOR *
 ******************************/

// Libraries
#include "opencv2/xfeatures2d.hpp"

// Constants
#define DEFAULT_OUTPUT_DIR "evaluation/"
#define DESCRIPTOR "SIFT"    // name of descriptor

void detectAndCompute(cv::Mat image, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors) {
	cv::Ptr<cv::Feature2D> f2d = cv::xfeatures2d::SIFT::create(); // use default parameters

	// detect keypoints
	f2d->detect(image, keypoints);

	// extract descriptors
	f2d->compute(image, keypoints, descriptors);
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

	for(auto const& fname : img_vec) {
		std::cout << "Extracting descriptors for " << fname << "\n";

		// Read the image
		cv::Mat img;
		img = cv::imread(fname, CV_LOAD_IMAGE_COLOR);

		std::string dt_fname;

		// get the type name (e.g. ref, e1, e2, etc.)
		std::vector<std::string> strs;
		boost::split(strs, fname, boost::is_any_of("/"));
		std::string img_name = strs.back();
		std::string seq_name = is_directory(input) ? strs[strs.size() - 2] : "";

		// parse out periods
		std::vector<std::string> strs_;
		boost::split(strs_, img_name, boost::is_any_of("."));
		std::string tp =  strs_[0];

		dt_fname = output_directory + descr_name + "/" + main_dir + seq_name + "/" + tp + ".csv";

		// Compute descriptors
		cv::Mat descriptors;
		std::vector<cv::KeyPoint> keypoints;
		detectAndCompute(img, keypoints, descriptors);

		// Open the descriptor file for saving
		std::ofstream f;
		f.open(dt_fname);
		f << cv::format(descriptors, cv::Formatter::FMT_CSV) << std::endl;
		f.close();
		std::cout << "Extraction complete. Stored at " << dt_fname << "\n";
	}
	return 0;
}

bool is_image(std::string fname) {
	std::string ext = boost::filesystem::extension(fname);
	return(ext == ".png" || ext == ".jpg" || ext == ".ppm" || ext == ".pgm");
}

bool is_hidden_file(std::string fname) {
    return(fname[0] == '.');
}
