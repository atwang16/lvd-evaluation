# Utils

This directory contains various project-specific libraries and a Python tool for formatting databases.

## detectors.hpp

`detectors.hpp` contains implementations of several keypoint detectors, each of which accepts as input an image and parameter file and returns a set of keypoints and corresponding affine transformations.

To add a detector to the library, add a prototype function to `detectors.hpp` and add the detector to `DETECTOR_MAP`. The function must have the form

```
void detector_name(Mat image, KeyPointCollection& kp_col, string parameter_file)
```

KeyPointCollection is a user-defined struct with the following definition:

```
struct KeyPointCollection {
	std::vector<cv::KeyPoint> keypoints;
	std::vector<cv::Mat> affine;
};
```

Include the implementation of the detector in `detectors.cpp`. Refer to other implementations for examples.

## descriptors.hpp

`descriptors.hpp` contains implementations of several descriptors, each of which accepts as input an image and corresponding set of keypoints and returns a set of descriptors, one for each keypoint.

To add a descriptor to the library, add a prototype function to `descriptors.hpp` and add the descriptor to `DESCRIPTOR_MAP`. The function must have the form

```
void descriptor(cv::Mat image, KeyPointCollection& kp_col, cv::Mat& descriptors, std::string parameter_file)
```

Include the implementation of the detector in `detectors.cpp`. Refer to other implementations for examples.


## utils.hpp

`utils.hpp` contains a variety of functions widely used throughout the project, such as a routine to parse .csv files and return OpenCV matrices.

## format_database.py

A utility to format image databases so that they are compatible with the projectexecutables. The tool assumes that the database is stored in the datasets folder, and that each sequence of corresponding images is stored in its own sequence folder within the database.

Usage: `python3 format_database.py database`

| Parameters | Description |
| ---------- | ----------- |
| `database` | the name of the database to be formatted |

The code contains two pre-processing lines, which allow the user to adjust the names of the sequences, images, and homography files as necessary (e.g. removing the first `n` characters from each name) before formatting them. To format the database correctly, sequences should originally be named with only their sequence names. Image labeling is based on sorting, and so provided that the images are in the correct order their original names will be ignored in formatting. The final format of the database should be as follows, assuming below that the database is named "database":

```
database
> dat_001_sequencename
>> dat_001_001.*
>> dat_001_002.*
>> dat_001_003.*
>> ...
>> H1to2p
>> H1to3p
>> ...
> dat_002_sequencename
>> dat_002_001.*
>> dat_002_002.*
>> dat_002_003.*
>> ...
>> H1to2p
>> H1to3p
>> ...
> dat_003_sequencename
>> dat_003_001.*
>> dat_003_002.*
>> dat_003_003.*
>> ...
>> H1to2p
>> H1to3p
>> ...
> ...
```

The final names of the sequences and images will be printed to console, and the user will be prompted to confirm that the formatting appears correct before continuing.

The currently supported file extensions are .jpg, .png, .ppm, and .pgm. Ground-truth homographies, labeled `H1toXp`, are always from image 1 to another image within the sequence.

