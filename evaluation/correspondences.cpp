/*
 * basetest.cpp
 *
 *  Created on: Jun 27, 2017
 *      Author: austin
 */

#include "utils.hpp"
#include <opencv2/opencv.hpp>
#include <boost/filesystem.hpp>

using namespace std;
using namespace cv;

#define PATH_DELIMITER "/"
#define BOOST_PATH_DELIMITER boost::is_any_of(PATH_DELIMITER)

int main(int argc, char *argv[]) {
	Mat kp_mat_1, kp_mat_2, homography;
	string results_file = "";
	vector<KeyPoint> kp_vec_1, kp_vec_2;
	vector< vector<int> > correct_matches;
	float kp_dist_thresh;
	int overwrite;

	if(argc < 7) {
		cout << "Usage ./correspondences keypoint_1 keypoint_2 homography kp_dist_thresh results_file overwrite_flag\n";
		return 1;
	}

	// Parse remaining arguments
	kp_mat_1 = parse_file(argv[1], ',');
	for(int i = 0; i < kp_mat_1.rows; i++) {
		KeyPoint kp = KeyPoint();
		kp.pt.x = kp_mat_1.at<float>(i, 0);
		kp.pt.y = kp_mat_1.at<float>(i, 1);
		kp_vec_1.push_back(kp);
	}

	kp_mat_2 = parse_file(argv[2], ',');
	for(int i = 0; i < kp_mat_2.rows; i++) {
		KeyPoint kp = KeyPoint();
		kp.pt.x = kp_mat_2.at<float>(i, 0);
		kp.pt.y = kp_mat_2.at<float>(i, 1);
		kp_vec_2.push_back(kp);
	}

	homography = parse_file(argv[3], ' ', CV_32F);
	kp_dist_thresh = stof(argv[4]);
	results_file = argv[5];
	overwrite = stoi(argv[6]);

	if(overwrite || !boost::filesystem::exists(results_file)) {
		// Count total number of correspondences
		int num_correspondences = 0;
		for(int i = 0; i < kp_vec_1.size(); i++) {
			for(int j = 0; j < kp_vec_2.size(); j++) {
				if(is_overlapping(kp_vec_1[i], kp_vec_2[j], homography, kp_dist_thresh)) {
					num_correspondences++;
					break;
				}
			}
		}

		cout << num_correspondences << " correspondences found.\n";

		ofstream f;
		f.open(results_file, ofstream::out);
		f << num_correspondences;
		f.close();
	}
	else {
		cout << "Number of correspondences already stored at " << results_file << "\n";
	}

	return 0;
}
