/*
 * fishervectors.cpp
 *
 *  Created on: Jul 17, 2017
 *      Author: austin
 */


extern "C" {
#include <vl/generic.h>
#include <vl/fisher.h>
#include <vl/gmm.h>
}

#include <iostream>
#include <cstdlib>
#include <opencv2/opencv.hpp>
#include <boost/filesystem.hpp>
#include <boost/algorithm/string.hpp>
#include <chrono>

using namespace std;
using namespace cv;
using namespace std::chrono;
using namespace boost::filesystem;

vector< vector<float> > parse_file(vl_size* num_descs, vl_size* dimension, string fname);
bool is_desc(std::string fname);
bool is_hidden_file(std::string fname);
string get_cat(string desc_fname);

int main(int argc, char *argv[]) {
	Mat query_desc, database_desc;
	string db_path, results = "";
	vector<KeyPoint> kp_vec_1, kp_vec_2;
	vector<string> desc_vec;
	vl_size max_em_iterations = 100, num_clusters = 100;

	if(argc < 4) {
		cout << "Usage ./fishervectors parameters_file desc_database results_folder\n";
		cout << "      ./fishervectors parameters_file desc_file results_folder\n";
		return 1;
	}

	// Load parameters for base test
	std::ifstream params(argv[1]);
	string line, var, value;
	std::vector<std::string> line_split;

	while(getline(params, line)) {
		boost::split(line_split, line, boost::is_any_of("="));
		var = line_split[0];
		value = line_split.back();
		if(var == "MAX_EM_ITERATIONS") {
			max_em_iterations = stoi(value);
		}
		else if(var == "NUM_CLUSTERS") {
			num_clusters = stoi(value);
		}
	}

	// Parse database path
	db_path = argv[2];

	results = argv[3];

	if(is_directory(db_path)) {
		vector<string> db_subdirs;

		// iterate through sequence folders of directory
		for(auto& entry : boost::make_iterator_range(directory_iterator(db_path), {})) {
			db_subdirs.push_back(entry.path().string()); // appends path of file to seqs vector
		}

		for(auto const& cat: db_subdirs) {
			vector<string> cat_split;
			if(is_directory(cat)) {
				for(auto& entry : boost::make_iterator_range(directory_iterator(cat), {})) {
					string fname = entry.path().string();
					if(is_desc(fname)) { // make sure file is a descriptor and that it is not the same file
						desc_vec.push_back(fname);
					}
				}
			}
		}
	}
	else if(is_desc(db_path)) {
		desc_vec.push_back(db_path);
	}
	else {
		cout << "Error: not a valid descriptor file. Make sure the naming convention is correct.\n";
		return 1;
	}

	// create a new instance of a GMM object for float data
	float *descs, *enc;
	vl_size num_descs, dimension;

	// Create fisher vectors
	for(int i = 0; i < desc_vec.size(); i++) {
		std::cout << "Extracting descriptors for " << desc_vec[i] << "\n";
		vector< vector<float> > vector_data = parse_file(&num_descs, &dimension, desc_vec[i]);
		descs = (float *)vl_malloc(sizeof(float) * (num_descs * dimension));

		// Place data in OpenCV matrix
		for(int des = 0; des < num_descs; des++) {
		   for(int dim = 0; dim < dimension; dim++) {
			   descs[des*dimension + dim] = vector_data[des][dim];
		   }
		}

		VlGMM* gmm = vl_gmm_new(VL_TYPE_FLOAT, dimension, num_clusters);
		vl_gmm_set_max_num_iterations(gmm, max_em_iterations);
		// set the initialization to random selection
		vl_gmm_set_initialization(gmm, VlGMMRand);
		// cluster the data, i.e. learn the GMM
		vl_gmm_cluster(gmm, descs, num_descs);
		// allocate space for the encoding
		enc = (float *)vl_malloc(sizeof(float) * 2 * dimension * num_clusters);
		// run fisher encoding
		vl_fisher_encode(enc, VL_TYPE_FLOAT, vl_gmm_get_means(gmm), dimension, num_clusters,
				vl_gmm_get_covariances(gmm), vl_gmm_get_priors(gmm), descs, num_descs, VL_FISHER_FLAG_IMPROVED);

		// save Fisher vector encoding
		vector<string> fname_split;
		boost::split(fname_split, desc_vec[i], boost::is_any_of("/,."));
		string img_name = fname_split[fname_split.size() - 2].substr(0, 11);
		string seq_name = is_directory(db_path) ? fname_split[fname_split.size() - 3] : "";
		string output_name = results + seq_name + "/" + img_name + "_fv.csv";
		std::ofstream f;
		f.open(output_name);
		for(int i = 0; i < 2 * dimension * num_clusters; i++) {
			f << enc[i] << "\n";
		}
		f.close();
		std::cout << "Fisher vector stored at " << output_name << "\n";

		vl_free(descs);
		vl_free(enc);
	}

	return 0;
}

bool is_hidden_file(string fname) {
    return(fname[0] == '.');
}

bool is_desc(string fname) {
	vector<string> fname_split;
	boost::split(fname_split, fname, boost::is_any_of("_,."));
	return(fname_split[fname_split.size() - 2] == "ds");
}

string get_cat(string fname) {
	vector<string> fname_split;
	boost::split(fname_split, fname, boost::is_any_of("_,."));
	int i = fname_split.size() - 5;
	return(fname_split[i] + "_" + fname_split[i + 1]);
}

vector< vector<float> > parse_file(vl_size* num_descs, vl_size* dimension, string fname) {
	std::ifstream inputfile(fname);
	string current_line;
	const char delimiter = ',';

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

	*num_descs = all_data.size();
	*dimension = all_data[0].size();

	return all_data;
}
