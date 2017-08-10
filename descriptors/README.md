# Keypoint and Descriptor Generation

## Generate Keypoints

### generate_keypoints.py

The Python interface code for generating keypoints from detectors. Applies the provided detector to an image database, locating files based on the paths stored in project_structure.txt. If a descriptor name is provided, parameters are extracted from the corresponding descriptor parameters file, and keypoints are stored in the folder of the corresponding descriptor; otherwise, the default values used to initialize the variables in the code are used, and keypoints will be stored in the folder of the corresponding detector. Keypoints are stored in .csv files in the results folder.
If the file is moved relative to the project root folder, where project_structure.txt is located, the path directory set in file for ROOT_PATH must be changed. The image database must also be formatted according to the standards laid out in the image formatting standards.

Usage: `python3 generate_keypoints.py detector database_name [descriptor_name] [-overwrite]`

| Parameters | Description |
| ---------- | ----------- |
| `detector` | the name of the keypoint detector which will be used to generate keypoints |
| `database_name` | the name of the image database to which the detector will be applied |
| `descriptor_name` | the name of the descriptor for which keypoints will be generated |
| `-overwrite` | a flag which controls whether previously existing keypoint files are overwritten or not |

Currently, the following detectors are supported:
* Difference of Gaussians (`dog`)
* Hessian Affine (`hesaff`)
* Hessian Laplace (`heslap`)
* Harris Affine (`haraff`)
* Harris Laplace (`harlap`)
* CenSurE (`censure`)
* MSER (`mser`)
* FAST (`fast`)
* AGAST (`agast`)
* SIFT detector (`sift`)
* BRISK detector (`brisk`)
* KAZE detector (`kaze`)
* ORB detector (`orb`)
* SURF detector (`surf`)
* U-SURF detector (`usurf`)
* AKAZE detector (`akaze`)

The first five detectors (`DoG`, `HesAff`, `HesLap`, `HarAff`, and `HarLap`) are implemented using the VlFeat library, and the remainder are implemented using OpenCV. To implement other detectors, one can either add implementations to `detectors.hpp` in the utils folder, or one can use other independent implementations, provided that the output data is identically formatted.

Keypoint files are .csv files, with one keypoint per line. The format of each keypoint is as follows:

```
x, y, size, angle, response, octave, class_id, a11, a12, a13, a21, a22, a23
```

The first 7 values correspond exactly to the parameters of an OpenCV KeyPoint object, and the latter six parameters form the corresponding affine matrix.

The `response`, `octave`, and `class_id` values are not crucial for non-OpenCV descriptors. By default `response` and `octave` are both initialized to 0, while `class_id` is initialized to -1.

A `time.csv` file will also be created in the results directory of the database, recording the time to generate each keypoint (in microseconds), the number of images processed, and the total number of keypoints, in that order.

### generate_keypoints.cpp

The underlying C++ code for generating keypoints. Running `generate_keypoints` directly allows for execution of the same routines without the assumptions of the wrapper code, since paths to each directory are passed directly to the executable.

Usage: `./generate_keypoints detector path_to_parameter_file image_dataset_root_folder path_to_destination overwrite_flag`

       `./generate_keypoints detector path_to_parameter_file path_to_image path_to_destination overwrite_flag`

| Parameters | Description |
| ---------- | ----------- |
| `detector` | the name of the keypoint detector which will be used to generate keypoints |
| `path_to_parameter_file` | path to the file which contains parameter values, or pass null to just use default values |
| `image_dataset_root_folder` | path to the directory containing the sequences of images |
| `path_to_image` | path to a single image |
| `path_to_destination` | path to the directory where results will be stored |
| `overwrite_flag` | 1 if previously existing files should be overwritten, or 0 otherwise |

The executable can either be run on an image database or a single image, as shown in the two use cases.

## Generate Descriptors

### generate_descriptors.py

The Python interface code for generating descriptors. Applies the provided descriptor to a database of keypoints, locating files based on the paths stored in `project_structure.txt`. Parameters are extracted from the corresponding descriptor parameters file, and descriptors are stored as .csv files in the folder of the corresponding descriptor.

Usage: `python3 generate_descriptors.py detector_name descriptor_name database_name [-overwrite]`

| Parameters | Description |
| ---------- | ----------- |
| `detector_name` | name of the detector whose keypoints will be used |
| `descriptor_name` | name of the descriptor |
| `database_name` | name of the image database |
| `-overwrite` | a flag which controls whether previously existing keypoint files are overwritten or not |

The detector name is used to locate the keypoints for the image database. If the names of detector `abc` and descriptor `xyz` are different, the results folder will be labeled `abc_xyz`.

Currently, the following detectors are supported:
* A-KAZE (`akaze`)
* BRIEF (`brief`)
* BRISK (`brisk`)
* CS-LBP (`cslbp`)
* FREAK (`freak`)
* KAZE (`kaze`)
* LATCH (`latch`)
* LIOP (`liop`)
* LUCID (`lucid`)
* ORB (`orb`)
* SIFT (`sift`)
* SURF (`surf`)
* U-SURF (`usurf`)

To implement other detectors, one can either add implementations to `descriptors.hpp` in the utils folder, or one can use other independent implementations, provided that it can read the keypoint data from the .csv files and output the results in an identical format.

Descriptors are stored in .csv files, with one descriptor per line.

The time to generate a descriptor for each keypoint, the number of images, and the number of keypoints will be recorded in a `time.csv` file.

### generate_descriptors.cpp

The underlying C++ code for generating descriptors. Running `generate_descriptors` directly allows for execution of the same routines without the assumptions of the wrapper code, since paths to each directory are passed directly to the executable.

Usage: `./generate_descriptors descriptor path_to_parameter_file image_dataset_root_folder keypoint_dataset_root_folder path_to_destination`

       `./generate_descriptors descriptor path_to_parameter_file path_to_image path_to_keypoint path_to_destination`

| Parameters | Description |
| ---------- | ----------- |
| `descriptor` | the name of the descriptor which will be used to generate keypoints |
| `path_to_parameter_file` | path to the file which contains parameter values |
| `image_dataset_root_folder` | path to the directory containing the sequences of images |
| `keypoint_dataset_root_folder` | path to the directory containing the keypoints of images |
| `path_to_image` | path to a single image |
| `path_to_keypoint` | path to a single corresponding keypoint file |
| `path_to_destination` | path to the directory where results will be stored |
| `overwrite_flag` | 1 if previously existing files should be overwritten, or 0 otherwise |

The executable can either be run on an image database or a single image, as shown in the two use cases.


