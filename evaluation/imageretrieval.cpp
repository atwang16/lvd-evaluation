/*
 * imageretrieval.cpp
 *
 *  Created on: Jul 6, 2017
 *      Author: austin
 */

#include <utils.hpp>
#include <boost/filesystem.hpp>
#include <boost/algorithm/string.hpp>
#include <chrono>

using namespace std;
using namespace cv;
using namespace std::chrono;
using namespace boost::filesystem;

#define PATH_DELIMITER "/"
#define BOOST_PATH_DELIMITER boost::is_any_of(PATH_DELIMITER)

bool is_fv(std::string fname);
bool is_hidden_file(std::string fname);
bool among_first_n_images(std::string fname, int n);
string get_cat(string desc_fname);
float cosine_similarity(Mat vec_1, Mat vec_2);
float euclidean_norm(Mat vec_1, Mat vec_2);

int main(int argc, char *argv[]) {
	Mat query_fv;
	string desc_name, fv_database_path, query_cat, results = "";
	int num_query_images = 1;

	if(argc < 5) {
		cout << "Usage ./imageretrieval2 desc_name query_fisher fisher_database num_query_images [results_file]\n";
		return -1;
	}

	// Parse remaining arguments
	desc_name = argv[1];

	string query_path = argv[2];
	query_fv = parse_file(query_path, ',');
	query_cat = get_cat(query_path);
	vector<string> query_path_split;
	boost::split(query_path_split, query_path, BOOST_PATH_DELIMITER);
	string query_name = query_path_split.back();

	// Extract all descriptors from database
	vector<string> fv_database;
	fv_database_path = argv[3];

	num_query_images = atoi(argv[4]);

	if(argc >= 6) {
		results = argv[5];
	}

	if(is_directory(fv_database_path)) {
		vector<string> db_subdirs;

		// iterate through sequence folders of directory
		for(auto& entry : boost::make_iterator_range(directory_iterator(fv_database_path), {})) {
			db_subdirs.push_back(entry.path().string()); // appends path of file to seqs vector
		}
		sort(db_subdirs.begin(), db_subdirs.end());

		for(auto const& cat: db_subdirs) {
			if(is_directory(cat)) {
				for(auto& entry : boost::make_iterator_range(directory_iterator(cat), {})) {
					string fname = entry.path().string();
					// make sure file is a descriptor and that it is not the same file, and exclude first image if requested
					if(is_fv(fname) && query_path != fname && !(among_first_n_images(fname, num_query_images))) {
						fv_database.push_back(fname);
					}
				}
			}
		}
	}

	// Calculate number of correspondences for each database image
	vector<float> distances;

	for(int i = 0; i < fv_database.size(); i++) {
		Mat db_fv = parse_file(fv_database[i], ',');

		distances.push_back(cosine_similarity(query_fv, db_fv)); // use cosine similarity to calculate distances between fisher vectors
	}

	// Find permutation of number of correspondences
	vector<int> ind;
	for(int i = 0; i < fv_database.size(); i++) {
		ind.push_back(i);
	}
	sort(ind.begin(), ind.end(),
			[&](const float& i, const float& j) {
        		return (distances[i] > distances[j]);
    		}
	);

	// Create vector of rank labels to account for ties
	vector<int> rank;
	rank.push_back(1); // initial value
	for(int ord = 2; ord <= ind.size(); ord++) {
		if(distances[ind[ord - 1]] == distances[ind[ord - 2]]) { // if equal number of matches is found
			rank.push_back(rank[rank.size() - 1]);
		}
		else {
			rank.push_back(ord);
		}
	}

	// Use data from last step to build metrics
	float ave_precision = 0;
	int num_rel_imgs = 0;
	bool is_first_img_rel = false;
	for(int ord = 1; ord <= ind.size(); ord++) {
		int i = ind[ord - 1];
		if(query_cat == get_cat(fv_database[i])) {
			num_rel_imgs++;
			ave_precision += (float)num_rel_imgs / rank[ord - 1];
			if(ord == 1) {
				is_first_img_rel = true;
			}
		}
	}
	ave_precision /= num_rel_imgs;

	// Export metrics
	if(results == "") {
		cout << desc_name << " descriptor results:\n";
		cout << "-----------------------"                                  << "\n";
		cout << "Average precision: " << ave_precision                     << "\n";
		cout << "Success:           " << (is_first_img_rel ? "yes" : "no") << "\n";
		cout                                                               << "\n";
		for(int ord = 1; ord <= ind.size(); ord++) {
			int i = ind[ord - 1];
			cout << setw(5) << right << rank[ord - 1] << " ";
			cout << setw(5) << right << distances[i] << " ";
			cout << fv_database[i] << "\n";
		}
	}
	else {
		std::ofstream f;

		f.open(results, std::ofstream::out | std::ofstream::app);
		f << query_name << "," << ave_precision << "," << (is_first_img_rel ? 1 : 0) << "\n";
		f.close();
	}

	return 0;
}

bool is_hidden_file(string fname) {
    return(fname[0] == '.');
}

bool is_fv(string fname) {
	vector<string> fname_split;
	boost::split(fname_split, fname, boost::is_any_of("_,."));
	return(fname_split[fname_split.size() - 2] == "fv");
}

bool among_first_n_images(std::string fname, int n) {
	vector<string> fname_split;
	boost::split(fname_split, fname, boost::is_any_of("_,."));
	return(stoi(fname_split[fname_split.size() - 3]) <= n);
}

string get_cat(string fname) {
	vector<string> fname_split;
	boost::split(fname_split, fname, boost::is_any_of("_,."));
	int i = fname_split.size() - 5;
	return(fname_split[i] + "_" + fname_split[i + 1]);
}

float euclidean_norm(Mat vec_1, Mat vec_2) {
	return norm(vec_1, vec_2, NORM_L2);
}

float cosine_similarity(Mat vec_1, Mat vec_2) {
     return vec_1.dot(vec_2) / (norm(vec_1) * norm(vec_2));
}
