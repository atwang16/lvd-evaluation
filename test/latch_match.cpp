/*
 * latch_match.cpp
 *
 *  Created on: Jul 5, 2017
 *      Author: austin
 */

#include <iostream>

#include "opencv2/opencv_modules.hpp"

#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>
#include <vector>

#ifdef HAVE_OPENCV_XFEATURES2D

// If you find this code useful, please add a reference to the following paper in your work:
// Gil Levi and Tal Hassner, "LATCH: Learned Arrangements of Three Patch Codes", arXiv preprint arXiv:1501.03719, 15 Jan. 2015

using namespace std;
using namespace cv;

const float inlier_threshold = 2.5f; // Distance threshold to identify inliers
const float nn_match_ratio = 0.8f;   // Nearest neighbor matching ratio

Mat parse_file(string fname, char delimiter, int type);

int main(void)
{
    Mat img1 = imread("/Users/austin/MIT/02_Spring_2017/MISTI_France/lvd-evaluation/datasets/Mikolajczyk/graf/img1.ppm", IMREAD_GRAYSCALE);
    Mat img2 = imread("/Users/austin/MIT/02_Spring_2017/MISTI_France/lvd-evaluation/datasets/Mikolajczyk/graf/img6.ppm", IMREAD_GRAYSCALE);

    Mat homography = parse_file("/Users/austin/MIT/02_Spring_2017/MISTI_France/lvd-evaluation/datasets/Mikolajczyk/graf/H1to6p", ' ', CV_32F);

    vector<KeyPoint> kpts1, kpts2;
    Mat desc1, desc2;

    Ptr<cv::ORB> orb_detector = cv::ORB::create(10000);

    Ptr<xfeatures2d::LATCH> latch = xfeatures2d::LATCH::create();

    orb_detector->detect(img1, kpts1);
    latch->compute(img1, kpts1, desc1);

    orb_detector->detect(img2, kpts2);
    latch->compute(img2, kpts2, desc2);

    BFMatcher matcher(NORM_HAMMING);
    vector< vector<DMatch> > nn_matches;
    matcher.knnMatch(desc1, desc2, nn_matches, 2);

    vector<KeyPoint> matched1, matched2, inliers1, inliers2;
    vector<DMatch> good_matches;
    for (size_t i = 0; i < nn_matches.size(); i++) {
        DMatch first = nn_matches[i][0];
        float dist1 = nn_matches[i][0].distance;
        float dist2 = nn_matches[i][1].distance;

        if (dist1 < nn_match_ratio * dist2) {
            matched1.push_back(kpts1[first.queryIdx]);
            matched2.push_back(kpts2[first.trainIdx]);
        }
    }

    for (unsigned i = 0; i < matched1.size(); i++) {
        Mat col = Mat::ones(3, 1, CV_32F);
        col.at<float>(0) = matched1[i].pt.x;
        col.at<float>(1) = matched1[i].pt.y;

        col = homography * col;
        col /= col.at<float>(2);
        double dist = sqrt(pow(col.at<float>(0) - matched2[i].pt.x, 2) +
            pow(col.at<float>(1) - matched2[i].pt.y, 2));

        if (dist < inlier_threshold) {
            int new_i = static_cast<int>(inliers1.size());
            inliers1.push_back(matched1[i]);
            inliers2.push_back(matched2[i]);
            good_matches.push_back(DMatch(new_i, new_i, 0));
        }
    }

//    Mat res;
//    drawMatches(img1, inliers1, img2, inliers2, good_matches, res);
//    imwrite("../results/test/latch_res.png", res);


    double inlier_ratio = inliers1.size() * 1.0 / matched1.size();
    cout << "LATCH Matching Results" << endl;
    cout << "*******************************" << endl;
    cout << "# Keypoints 1:                        \t" << kpts1.size() << endl;
    cout << "# Keypoints 2:                        \t" << kpts2.size() << endl;
    cout << "# Matches:                            \t" << matched1.size() << endl; // number of matches
    cout << "# Inliers:                            \t" << inliers1.size() << endl; // number of correct matches
    cout << "# Inliers Ratio:                      \t" << inlier_ratio << endl; // proportion of correct matches
    cout << endl;

    return 0;
}

#else

int main()
{
    std::cerr << "OpenCV was built without xfeatures2d module" << std::endl;
    return 0;
}

#endif

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
						values.push_back(stof(single_value));
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
		vector< vector<int> > all_data;

		// read each line
		while(getline(inputfile, current_line)) {
			if(current_line != "") {
				vector<int> values;
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
			  vect.at<int>(row, col) = all_data[row][col];
		   }
		}
		return vect;
	}
}
