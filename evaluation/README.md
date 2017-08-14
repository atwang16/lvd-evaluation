# Descriptor Evaluation

## Base Test

### correspondences.py

The Python interface code for computing the correct correspondences between keypoints for an image database. The results are used in `basetest` to measure the effectiveness of descriptors in identifying correct matches between keypoints.

Usage: `python3 correspondences.py detector_name database_name

| Parameters | Description |
| ---------- | ----------- |
| `descriptor_name` | the name of the descriptor |
| `database_name` | the name of the database |

Results are stored in the same location as keypoints, with the suffix `co.csv`. Each row corresponds to the index of the keypoint of image 1 in the sequence, and the indices listed in each row indicate matching keypoints of the image given by the filename. For keypoints in image 1 without a corresponding keypoint, `-1` is used to indicate that no matches were found.

### correspondences.cpp

The underlying C++ code for running correspondences. Running correspondences directly allows for execution of the same routines without the assumptions of the wrapper code, since paths to each directory and parameters are passed directly to the executable, on a single pair of images rather than a complete database.

Usage: `./correspondences keypoint_1 keypoint_2 homography kp_dist_thresh results_file`

| Parameters | Description |
| ---------- | ----------- |
| `keypoint_1` | path to file of keypoints for image 1 |
| `keypoint_2` | path to file of keypoints for image 2 |
| `homography` | path to file containing ground-truth homography from image 1 to 2 |
| `kp_dist_thresh` | threshold for the distance, in pixels, between a projected keypoint of image 1 and a keypoint of image 2 which determines when keypoints are said to correspond to the same point |
| `results_file` | path to file where results of correspondence computations will be saved |

### basetest.py

The Python interface code for applying the base test to descriptors of a database, evaluating the ability of the descriptor to compute invariant descriptions of keypoints and produce a high proportion of correct matches between keypoints of two images. Evaluates descriptors based on comparison of keypoint matching to ground truth homographies. For each sequence of a database, the descriptors of the first image are matched to those of each other image in the sequence using a brute force matching scheme with the associated distance metric between descriptors. The matches are further refined by applying a distance ratio threshold, and the remaining matches are compared to the ground-truth homography to determine how many of those matches are correct matches. Results are outputted to a .csv file, and the matching ratio, matching score, precision, and recall are calculated across all comparisons.
If the file is moved relative to the project root folder, where project_structure.txt is located, the path directory set in file for `ROOT_PATH` must be changed. 

Usage: `python3 basetest.py descriptor_name database_name [-results_only]`

| Parameters | Description |
| ---------- | ----------- |
| `descriptor_name` | the name of the descriptor |
| `database_name` | the name of the database |
| `-results_only` | skip running the base test and only process the prior recorded results |

Parameters for `basetest.py` are extracted from `basetest_parameters.txt`, located in the same directory:

| Parameters | Description |
| ---------- | ----------- |
| `DIST_RATIO_THRESH` | the threshold for the ratio of distances of the first nearest neighbor to the second nearest neighbor for a keypoint match |
| `KP_DIST_THRESH` | the threshold in pixels for determining whether two keypoints are matching |
| `CAP_CORRECT_DISPLAYED` | a flag which controls whether the number of keypoint matches displayed on the output image is capped |
| `NB_KP_TO_DISPLAY` | the number of keypoint matches to display on the output image, if `CAP_CORRECT_DISPLAYED=1` |

From the parameters file of the specific descriptor for which the base test is being run, the program will search for `DISTANCE` to set the appropriate distance metric (`L1`, `L2`, `HAMMING`, or `HAMMING2`). The default is `L2`.

The base test stores the majority of results in `desc_dat_basetest.csv`, where `desc` is the name of the descriptor and `dat` is the first three letters of the database. Each row corresponds to one execution of the base test, with the following format:
* image name 1
* image name 2
* number of keypoints in image 1
* number of keypoints in image 2
* number of matches (after the distance ratio test)
* number of correct matches, according to the ground truth homography
* number of correct correspondences between two images
* match ratio
* matching score
* precision
* recall
* time to compute all matches (in milliseconds)

Refer to the paper for elaboration on the definition of each evaluation metric.

For each comparison, images are also created which provide a visual for the matches between keypoints. Since the number of correct matches can be fairly large, only a subset of the matches are shown to provide a visual confirmation for descriptor performance.

### basetest.cpp

The underlying C++ code for running the base test. Running basetest directly allows for execution of the same routines without the assumptions of the wrapper code, since paths to each directory and parameters are passed directly to the executable. Furthermore, the basetest executable is designed to be run on a single pair of images rather than a complete database.

Usage: `./basetest parameters_file descriptor_name img_1 desc_1 keypoint_1 img_2 desc_2 keypoint_2 homography dist_metric [-s stat_results] [-d draw_results]`

| Parameters | Description |
| ---------- | ----------- |
| `parameters_file` | path to a text file with all of the parameters |
| `descriptor_name` | name of descriptor |
| `img_1` | path to image 1 |
| `desc_1` | path to file with descriptors for image 1 |
| `keypoint_1` | path to file with keypoints for image 1 |
| `img_2` | path to image 2 |
| `desc_2` | path to file with descriptors for image 2 |
| `keypoint_2` | path to file with keypoints for image 2 |
| `homography` | path to file with ground-truth homography |
| `dist_metric` | name of distance metric used to compare descriptors |
| `stat_results` | path to file where statistics of each basetest execution will be stored; must be proceeded with `-s` |
| `draw_results` | path to file where image matching output will be stored; must be proceeded with `-d` |

Note that `stat_results` and `draw_results` are optional. If `stat_results` is not provided, the statistics for the match will be outputted directly to console and not saved.

## Image Retrieval

### fishervectors.py

Computes a bag-of-words dictionary based on a training subset of the description in the target database using a Gaussian mixture model. Computes the Fisher vector for each image based on the visual dictionary and extracted descriptors. fishervectors.py is the Python interface code for fishervectors.cpp, which makes it easier to generate the directory paths for the underlying C++ code.

Usage: `python3 fishervectors.py descriptor_name database_name`

| Parameters | Description |
| ---------- | ----------- |
| `descriptor_name` | name of descriptor |
| `database_name` | name of database |

The visual dictionary for the given descriptor is stored in `descriptor_visual_dictionary.csv` and can be used to compute the Fisher vectors for other datasets without needing to train new clusters. Parameters for the `fishervectors.py` routine are stored in `fisher_parameters.txt`.

| Parameters | Description |
| ---------- | ----------- |
| `MAX_EM_ITERATIONS` | Maximum number of EM iterations in computation of Gaussian mixture model |
| `NUM_CLUSTERS` | The number of clusters in the Gaussian mixture model |
| `NUMBER_DESCRIPTORS_TO_SAMPLE` | The number of descriptor vectors used in the training set |
| `FIRST_TRAINING_SEQUENCE` | The ID of the first image sequence used in the training set |
| `NUMBER_OF_TRAINING_SEQUENCES` | The number of image sequences used in the training set, starting from `FIRST_TRAINING_SEQUENCE`. If 0, all sequences will be used starting from `FIRST_TRAINING_SEQUENCE`. |

### fishervectors.cpp

The underlying C++ code for computing Fisher vectors. Running `fishervectors` directly allows for execution of the same routines without the assumptions of the wrapper code, since paths to each directory and parameters are passed directly to the executable. The `fishervectors` executable can also be run on a single pair of images rather than a complete database.

Usage: `./fishervectors parameters_file desc_database results_folder [load_dictionary]`

       `./fishervectors parameters_file desc_file results_folder [load_dictionary]`

| Parameters | Description |
| ---------- | ----------- |
| `parameters_file` | path to file with parameters |
| `desc_database` | path to folder with descriptors |
| `desc_file` | path to a file with descriptors for a single image |
| `results_folder` | path to folder where results will be stored |
| `load_dictionary` | if provided, the visual dictionary will be loaded from here if the file exists or saved there after creating it |

### imageretrieval.py

Runs an image retrieval test on a specified dataset, using the associated Fisher vectors and cosine similarity to determine a similarity metric between images. For each query image, the distance is measured to the database images and rankedto produce mean average precision and success rate metrics. `imageretrieval.py` is the Python interface code for `imageretrieval.cpp`, which makes it easier to generate the directory paths for the underlying C++ code.

Usage: `python3 imageretrieval.py descriptor_name database [-generate_fishervectors] [-results_only]`

| Parameters | Description |
| ---------- | ----------- |
| `descriptor_name` | name of the descriptor |
| `database` | name of the database |
| `-generate_fishervectors` | generate Fisher vectors for the database |
| `-results_only` | skip running the base test and only process the prior recorded results |

The parameters for `imageretrieval.py` are stored in `imageretrieval_parameters.txt`:

| Parameters | Description |
| ---------- | ----------- |
| `QUERY_SAMPLE_SIZE` | the number of queries used in the image retrieval test for each sequence |

If the query size is 0, then every image will be used as a query. Otherwise, the first `QUERY_SAMPLE_SIZE` images will be used as queries, and the remainder of the images will comprise the search database.

Results for each query, namely the average precision and success rate, are stored in `descriptor_mik_imageretrieval2.csv`, with the following format:
```
query, average precision, success
```

Refer to the paper for definitions of the evaluation metrics.

### imageretrieval.cpp

The underlying C++ code for the image retrieval test. Running `imageretrieval` directly allows for execution of the same routines without the assumptions of the wrapper code, since paths to each directory are passed directly to the executables.

Usage: `./imageretrieval2 descriptor_name query_fisher fisher_database num_query_images [results_file]`

| Parameters | Description |
| ---------- | ----------- |
| `descriptor_name` | name of descriptor |
| `query_fisher` | path to the Fisher vector for the query image |
| `fisher_database` | path to the database of Fisher vectors |
| `num_query_images` | number of query images to exclude from beginning of each sequence |
| `results_file` | path to file where results will be stored |


