# lvd-evaluation
Source code, evaluation framework, and datasets for "A Comparative Study of Local Visual Descriptors," researched at ESIEE during the summer of 2017.

## Directory Structure and Naming Conventions
- datasets - directory containing all image datasets used in tests (not included in repository)
- descriptors - source code and parameter files for local visual descriptors
- evaluation - source code for evaluation routines (calculating distance thresholds, base test, application test)
- results - directory containing keypoints and descriptor files and all files outputted by evaluation tests (not included in repository)
- utils - directory containing the utils.hpp library and other auxiliary utilities

### Datasets
Datasets are not included in the github repository due to database size, but all databases used in this study  can be publicly downloaded online. A Python utility is provided in the utils folder (`format_database.py`) to rename subdirectories and images in a downloaded database to be compatible with the Python interface files provided in the repository. The datasets used for this study include the following:
- Amsterdam Library of Objects (ALOI)
- Caltech-256 dataset
- DaLI
- HPatches
- Iguazu
- Mikolajczyk et. al.

The naming convention and directory structure for image files must be as follows in order for Python code to run properly. As noted below, the C++ executables are designed to be run independently from any directory structure, for small-scale testing and use of the test programs here. For larger-scale testing, it is recommended to set up the repository as indicated here (specifically the datasets) and to use the Python interface code to automatically generate descriptors and test results.

Each dataset should be stored in a separate directory within the `datasets` folder, labeled with its name (recommended at least three letters). Each dataset contains a folder for each sequence of corresponding images, labeled `dat_sss_sequence`, where `dat` is the first three letters of the dataset, `nnn` is a three-digit number which uniquely identifies each sequence within the dataset, and `sequence` is the name of the sequence. Images for each sequence are labeled as `dat_nnn_iii.ext` (where .ext is the image extension), where, in addition to the prior conventions, `iii` is a three-digit number which uniquely identifies each image. In particular, image 001, in base tests, is always the reference image to which the rest are compared. Sequence folders also store the ground-truth homography files, named as `H1to#p`. For instance, the convention applied to Mikolajczyk would be as follows:
- mikolajczyk
  - mik_001_bark
    - mik_001_001.ppm
    - mik_001_002.ppm
    - ...
    - mik_001_006.ppm
    - H1to2p
    - H1to3p
    - ...
    - H1to6p
  - mik_002_bikes
    - mik_002_001.ppm
    - ...
  - mik_003_boat
  - mik_004_graf
  - mik_005_leuven
  - mik_006_trees
  - mik_007_ubc
  - mik_008_wall

The `format_database.py` Python utility has been provided to quickly format datasets into the above form. An example usage is shown here:
```
python format_database.py path_to_database
```

### descriptors

`descriptors` contains two types of files. The first are C++ source files for each of the local visual descriptors, which when executed produce .csv files for the extracted keypoints and descriptors. Each is labeled with its descriptor name (e.g. sift.cpp, latch.cpp, etc.). The other files are .txt files which store values for the parameters of hthe descriptors (such as the number of keypoints to produce). These files allow for easy modification of descriptor parameters without requiring one to rebuild the source code each time. All parameter files in this repository associate a keyword with a value using the `=` delimeter.

### evaluation

The evaluation folder contains the source code for distance threshold computation, the base test, and the application test (image retrieval). Each has a C++ source file and a Python script. The C++ files are useful for small-scale testing, as they take direct inputs for all of the file inputs and so are easy to run independent of any directory structure. Most of the C++ files are designed to run on a single instance of the test, such as a single pair of images for the base test. The Python files assume that the directory structure matches that described here and makes it easy to run the C++ executables on an entire database of images and generate the appropriate arguments. It is recommended to use these for any large scale descriptor testing, as it will make it more efficient to generate the results.

### results

The results folder contains all of the descriptors and keypoints extracted from images, in addition to all of the results for the various tests performed in the study. The structure of the `results` folder is similar to the `datasets folder, except that each dataset is stored in a folder for the descriptor for which it was computed. Keypoint and descriptor files have a similar name convention to images, except that descriptors end in `_ds.csv` and keypoint files end in `_kp.csv`.

The descriptor executables referenced above will, in addition to keypoint and descriptor files, generate a `timeresults.txt` file within the corresponding image dataset folder, which will record the amount of time it took to process all of the descriptors.

The base test will generate two files for each pair of images: a .txt file with the statistics of the test (e.g. precision and recall values), and a .png file with a sample of correct correspondences between the two images. These will be stored in their respective sequence folders, along with the descriptors and keypoints in the test.

The image retrieval test will generate a running .csv file of all of the trials, with the average precision and an indicator for the trial's success for each query. The file is likewise stored directly in the dataset folder for the corresponding descriptor.

Lastly, the data for distances to descriptors is stored directly in the corresponding descriptor folder, including .csv files for all correctly corresponding distances, all correct distances found (positive distances), and all closest non-corresponding distances (negative distances). A graph is also generated which aggregates all of the data into a presentable format.

## How to Run Files

The base test, application test, and distance threshold Python programs are all run by passing the name of the descriptor and database as arguments, in that order, to the programs as command-line arguments. For the distance threshold calculation, an optional argument `-append` can be passed at the end to prevent overwriting of past data collected. An example is shown below:
```
python basetest.py sift mikolajczyk
```
