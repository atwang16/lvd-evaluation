# Evaluation Code

## Base Test

### basetest.py

Usage: `python3 basetest.py descriptor_name database_name [-results_only]`

| -results_only | skip running the base test and only process the prior recorded results |

## Image Retrieval

### fishervectors.py

Computes a bag-of-words dictionary based on a training subset of the descriptors
in the target database, if one does not already exist, using a Gaussian mixture model. Computes the Fisher vector for each image based on the visual dictionary and extracted descriptors. fishervectors.py is the Python interface code for fishervectors.cpp, which makes it easier to generate the directory paths for the underlying C++ code.

Usage: `python3 fishervectors.py descriptor_name database_name

Parameters for the fishervectors.py routine are stored in fisher_parameters.txt.

| **Parameters** | **Description** |
| -------------- | --------------- |
| MAX_EM_ITERATIONS | Maximum number of EM iterations in computation of Gaussian mixture model |
| NUM_CLUSTERS | The number of clusters in the Gaussian mixture model |
| NUMBER_DESCRIPTORS_TO_SAMPLE | The number of descriptor vectors used in the training set |
| FIRST_TRAINING_SEQUENCE | The ID of the first image sequence used in the training set |
| NUMBER_OF_TRAINING_SEQUENCES | The number of image sequences used in the training set, starting from FIRST_TRAINING_SEQUENCE. If 0, all sequences will be used starting from FIRST_TRAINING_SEQUENCE. |

### imageretrieval2.py

Runs an image retrieval test on a specified dataset, using the associated Fisher vectors and cosine similarity to determine a similarity metric between images. For each query image, the distance is measured to the database images and rankedto produce mean average precision and success rate metrics.

Usage: `python3 imageretrieval2.py descriptor_name database [-generate_fishervectors] [-results_only]`

| -generate_fishervectors | generate Fisher vectors for the database |
| -results_only | skip running the base test and only process the prior recorded results |
