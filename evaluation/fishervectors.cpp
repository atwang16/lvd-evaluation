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

#include "utils.hpp"
#include <cstdlib>
#include <boost/filesystem.hpp>
#include <boost/algorithm/string.hpp>
#include <chrono>
#include <ctime>

using namespace std;
using namespace cv;
using namespace std::chrono;
using namespace boost::filesystem;

#define PATH_DELIMITER "/"
#define BOOST_PATH_DELIMITER boost::is_any_of(PATH_DELIMITER)

bool prob(float p);
bool is_desc(std::string fname);
bool is_hidden_file(std::string fname);
string get_cat(string desc_fname);

enum data {MEANS, COVARIANCES, PRIORS, END};

int main(int argc, char *argv[]) {
	string desc_name, db_path, results = "", dictionary = "";
	vector<string> desc_vec;
	vl_size max_em_iterations = 100, num_clusters = 256;
	int sample_size = 10000, first_training_sequence = 1, num_training_sequences = 0, first_train_ind = 0, num_train_imgs = 0, i = 1;;
	float *means, *covariances, *priors;
	vector<float> means_vec, covariances_vec, priors_vec;

	srand (time(NULL));

	if(argc < 4) {
		cout << "Usage ./fishervectors parameters_file desc_database results_folder [load_dictionary]\n";
		cout << "      ./fishervectors parameters_file desc_file results_folder [load_dictionary]\n";
		return 1;
	}

	// Load parameters for base test
	std::ifstream params(argv[1]);
	string line, var, value;
	std::vector<std::string> line_split;

	while(getline(params, line)) {
		boost::split(line_split, line, boost::is_any_of("="));
		var = line_split[0];
		value = line_split[1];
		if(var == "MAX_EM_ITERATIONS") {
			max_em_iterations = stoi(value);
		}
		else if(var == "NUM_CLUSTERS") {
			num_clusters = stoi(value);
		}
		else if(var == "NUMBER_DESCRIPTORS_TO_SAMPLE") {
			sample_size = stoi(value);
		}
		else if(var == "FIRST_TRAINING_SEQUENCE") {
			first_training_sequence = stoi(value);
		}
		else if(var == "NUM_TRAINING_SEQUENCES") {
			num_training_sequences = stoi(value);
		}
	}

	db_path = argv[2];
	results = argv[3];
	if(argc >= 5) {
		dictionary = argv[4];
	}

	if(is_directory(db_path)) {
		vector<string> db_subdirs;

		// iterate through sequence folders of directory
		for(auto& entry : boost::make_iterator_range(directory_iterator(db_path), {})) {
			db_subdirs.push_back(entry.path().string()); // appends path of file to seqs vector
		}
		sort(db_subdirs.begin(), db_subdirs.end());

		for(auto const& cat: db_subdirs) {
			vector<string> cat_split;
			if(is_directory(cat)) {
				for(auto& entry : boost::make_iterator_range(directory_iterator(cat), {})) {
					string fname = entry.path().string();
					if(is_desc(fname)) { // make sure file is a descriptor and that it is not the same file
						if(i == first_training_sequence) {
							first_train_ind = desc_vec.size();
						}
						desc_vec.push_back(fname);
						if(i >= first_training_sequence && (i < first_training_sequence + num_training_sequences || num_training_sequences == 0)) {
							num_train_imgs++;
						}
					}
				}
			}
			i++;
		}
		sort(desc_vec.begin(), desc_vec.end());
	}
	else if(is_desc(db_path)) {
		desc_vec.push_back(db_path);
	}
	else {
		cout << "Error: " << db_path << " is not a valid database directory or descriptor file. Make sure the naming convention is correct.\n";
		return 1;
	}

	// create a new instance of a GMM object for float data
	vector< Mat > desc_vec_mat, desc_vec_train_mat;
	float *descs, *enc, *d_ptr;
	vl_size num_descs = 0, dimension = 0;

	// Amalgamate descriptors into one
	cout << "Loading data...\n";
	for(int i = 0; i < desc_vec.size(); i++) {
		Mat m = parse_file(desc_vec[i], ',');
		m.convertTo(m, CV_32F);
		desc_vec_mat.push_back(m);
		if(first_train_ind <= i && i < first_train_ind + num_train_imgs) {
			desc_vec_train_mat.push_back(m);
		}
		num_descs += desc_vec_mat.back().rows;
	}

	cout << num_descs << " descriptors loaded.\n";

	dimension = desc_vec_mat.back().cols;

	if(dictionary == "" || !exists(dictionary)) { // no dictionary provided
		descs = (float *)vl_malloc(sizeof(float) * (sample_size * dimension));
		d_ptr = descs;
		int descs_remaining = num_descs, to_sample = sample_size;
		for(int i = 0; to_sample > 0 && i < desc_vec_train_mat.size(); i++) {
			Mat m = desc_vec_train_mat[i];
			for(int j = 0; to_sample > 0 && j < m.rows; j++) {
				if(prob(to_sample / descs_remaining)) {
					memcpy(d_ptr, m.row(j).data, dimension * sizeof(float));
					d_ptr += dimension;
					to_sample--;
				}
				descs_remaining--;
			}
		}

		cout << sample_size << " descriptors sampled.\n";

		// Create visual dictionary
		cout << "Constructing visual dictionary...\n";
		VlGMM* gmm = vl_gmm_new(VL_TYPE_FLOAT, dimension, num_clusters);
		vl_gmm_set_max_num_iterations(gmm, max_em_iterations);
		// set the initialization to random selection
		vl_gmm_set_initialization(gmm, VlGMMRand);
		// cluster the data, i.e. learn the GMM
		vl_gmm_cluster(gmm, descs, sample_size);

		means = (float *)vl_gmm_get_means(gmm);
		covariances = (float *)vl_gmm_get_covariances(gmm);
		priors = (float *)vl_gmm_get_priors(gmm);

		// store in dictionary
		if(dictionary != "") {
			std::ofstream f;
			f.open(dictionary);

			for(int i = 0; i < dimension * num_clusters; i++) {
				f << means[i] << ",";
			}
			f << "\n";

			for(int i = 0; i < dimension * num_clusters; i++) {
				f << covariances[i] << ",";
			}
			f << "\n";

			for(int i = 0; i < num_clusters; i++) {
				f << priors[i] << ",";
			}
			f << "\n";

			f.close();
		}
		vl_free(descs);
	}
	else { // load from dictionary
		std::ifstream inputfile(dictionary);
		string current_line;
		int index = 0;

		cout << "Loading visual dictionary from file...\n";

		// read means
		while(getline(inputfile, current_line) && index < END) {
			if(current_line != "") {
				stringstream str_stream(current_line);
				string single_value;

				// Read each value with delimiter
				while(getline(str_stream, single_value, ',')) {
					if(single_value != "") {
						switch(index) {
						case MEANS:
							means_vec.push_back(stof(single_value));
							break;

						case COVARIANCES:
							covariances_vec.push_back(stof(single_value));
							break;

						case PRIORS:
							priors_vec.push_back(stof(single_value));
							break;

						default:
							break;
						}
					}
				}
			}
			index++;
		}
		if(means_vec.size() != dimension * num_clusters || covariances_vec.size() != dimension * num_clusters
				|| priors_vec.size() != num_clusters || index < END) {
			cout << "Error: insufficient amount of data stored in dictionary. Aborting...\n";
			return 1;
		}

		means = (float *)means_vec.data();
		covariances = (float *)covariances_vec.data();
		priors = (float *)priors_vec.data();

		inputfile.close();
	}

	cout << "***\n";

	// Create fisher vectors
	vl_size enc_size = 2 * dimension * num_clusters;
	for(int i = 0; i < desc_vec.size(); i++) {
		std::cout << "Extracting fisher vectors for " << desc_vec[i] << "\n";

		// allocate space for the encoding
		enc = (float *)vl_malloc(sizeof(float) * enc_size);
		// run fisher encoding
		vl_fisher_encode(enc, VL_TYPE_FLOAT, (const void *)means, dimension, num_clusters, (const void *)covariances,
				(const void *)priors, desc_vec_mat[i].data, desc_vec_mat[i].rows, VL_FISHER_FLAG_IMPROVED);

		// save Fisher vector encoding
		vector<string> fname_split;
		boost::split(fname_split, desc_vec[i], boost::is_any_of("/."));
		string img_name = fname_split[fname_split.size() - 2].substr(0, 11);
		string seq_name = is_directory(db_path) ? (fname_split[fname_split.size() - 3] + PATH_DELIMITER) : "";
		string output_name = results + PATH_DELIMITER + seq_name + img_name + "_fv.csv";
		std::ofstream f;
		f.open(output_name);
		for(int i = 0; i < enc_size; i++) {
			f << enc[i] << "\n";
		}
		f.close();
		std::cout << "Fisher vector stored at " << output_name << "\n\n";

		vl_free(enc);
	}

	return 0;
}

bool prob(float p) {
	return p > 0.0 && (p >= 1.0 || rand() < p * RAND_MAX);
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
