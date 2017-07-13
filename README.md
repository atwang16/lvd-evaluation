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

Each dataset should be stored in a separate directory within the `datasets` folder, labeled with its name (recommended at least three letters). Each dataset contains a folder for each sequence of corresponding images, labeled `dat_sss_sequence`, where `dat` is the first three letters of the dataset, `nnn` is a three-digit number which uniquely identifies each sequence within the dataset, and `sequence' is the name of the sequence. Images for each sequence are labeled as `dat_nnn_iii.ext` (where .ext is the image extension), where, in addition to the prior conventions, `iii` is a three-digit number which uniquely identifies each image. In particular, image 001, in base tests, is always the reference image to which the rest are compared. Sequence folders also store the ground-truth homography files, named as `H1to#p`. For instance, the convention applied to Mikolajczyk would be as follows:
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

where
- "database" is the name of the database, and "dat" is the first three letters of the database
- "sequence" is the name of the image sequence, and "nnn" is a three-digit integer (padded with zeros) associated with each image sequence
- "iii" is the three-digit integer (padded with zeros) associated with each image
- The "ds" suffix marks csv files which contain descriptors for the named image
- The "kp" suffix marks csv files which contain keypoints for the named image

### Extracting Descriptors

The `build_database.py` Python file has been provided to allow for easily extracting keypoints and descriptors for a given local visual descriptor. (to be continued)
