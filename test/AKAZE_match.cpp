#include <opencv2/features2d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>
#include <vector>
#include <iostream>

using namespace std;
using namespace cv;

const float inlier_threshold = 2.5f; // Distance threshold to identify inliers
const float nn_match_ratio = 0.8f;   // Nearest neighbor matching ratio

Mat parse_file(string fname, char delimiter, int type);

int main(void)
{
	Mat img1 = imread("/Users/austin/MIT/02_Spring_2017/MISTI_France/lvd-evaluation/datasets/Mikolajczyk/bikes/img1.ppm", IMREAD_GRAYSCALE);
	Mat img2 = imread("/Users/austin/MIT/02_Spring_2017/MISTI_France/lvd-evaluation/datasets/Mikolajczyk/bikes/img4.ppm", IMREAD_GRAYSCALE);

	Mat homography = parse_file("/Users/austin/MIT/02_Spring_2017/MISTI_France/lvd-evaluation/datasets/Mikolajczyk/bikes/H1to4p", ' ', CV_32F);

    vector<KeyPoint> kpts1, kpts2;
    Mat desc1, desc2;

    Ptr<AKAZE> akaze = AKAZE::create();
    akaze->detectAndCompute(img1, noArray(), kpts1, desc1);
    akaze->detectAndCompute(img2, noArray(), kpts2, desc2);

    BFMatcher matcher(NORM_HAMMING);
    vector< vector<DMatch> > nn_matches;
    matcher.knnMatch(desc1, desc2, nn_matches, 2);

    vector<KeyPoint> matched1, matched2, inliers1, inliers2;
    vector<DMatch> good_matches;
    for(size_t i = 0; i < nn_matches.size(); i++) {
        DMatch first = nn_matches[i][0];
        float dist1 = nn_matches[i][0].distance;
        float dist2 = nn_matches[i][1].distance;

        if(dist1 < nn_match_ratio * dist2) {
            matched1.push_back(kpts1[first.queryIdx]);
            matched2.push_back(kpts2[first.trainIdx]);
        }
    }

    for(unsigned i = 0; i < matched1.size(); i++) {
        Mat col = Mat::ones(3, 1, CV_32F);
        col.at<float>(0) = matched1[i].pt.x;
        col.at<float>(1) = matched1[i].pt.y;

        col = homography * col;
        col /= col.at<float>(2);
        float dist = sqrt( pow(col.at<float>(0) - matched2[i].pt.x, 2) +
                            pow(col.at<float>(1) - matched2[i].pt.y, 2));

        if(dist < inlier_threshold) {
            int new_i = static_cast<int>(inliers1.size());
            inliers1.push_back(matched1[i]);
            inliers2.push_back(matched2[i]);
            good_matches.push_back(DMatch(new_i, new_i, 0));
        }
    }

//    Mat res;
//    drawMatches(img1, inliers1, img2, inliers2, good_matches, res);
//    imwrite("res.png", res);

    double inlier_ratio = inliers1.size() * 1.0 / matched1.size();
    cout << "A-KAZE Matching Results" << endl;
    cout << "*******************************" << endl;
    cout << "# Keypoints 1:                        \t" << kpts1.size() << endl;
    cout << "# Keypoints 2:                        \t" << kpts2.size() << endl;
    cout << "# Matches:                            \t" << matched1.size() << endl;
    cout << "# Inliers:                            \t" << inliers1.size() << endl;
    cout << "# Inliers Ratio:                      \t" << inlier_ratio << endl;
    cout << endl;

    return 0;
}

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
