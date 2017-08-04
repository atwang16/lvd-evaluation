import subprocess
import sys
import os

ROOT_PATH = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
results_path = None
image_all_db_path = None
image_db_path = None
descriptors_parameter_file = None
keypoint_executable = None
desc_name = None

def generate_keypoints(det, desc, db):
    global keypoint_executable, descriptors_parameter_file, image_db_path, results_path
    args = [keypoint_executable, det, descriptors_parameter_file, image_db_path, results_path]
    if desc is not None:
        print(det + ": " + "Extracting keypoints from " + db)
    else:
        print(det + "Extracting keypoints for " + desc + " from " + db)
    print("***")
    subprocess.run(args)


# MAIN #

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python build_keypoints.py detector database_name [descriptor_name]")
        sys.exit(1)

    detector = sys.argv[1]
    database = sys.argv[2]

    if len(sys.argv) >= 4:
        desc_name = sys.argv[3]
    else:
        desc_name = detector.lower()
        descriptors_parameter_file = "null"

    with open(os.path.join(ROOT_PATH, "project_structure.txt")) as f:
        line = f.readline()
        while line != "":
            line_split = line.split("=")
            var = line_split[0]
            directory = line_split[-1]
            if directory[-1] == "\n":
                directory = directory[:-1]
            if var == "RESULTS_FOLDER":
                results_path = os.path.join(ROOT_PATH, directory)
            elif var == "DATASETS_FOLDER":
                image_all_db_path = os.path.join(ROOT_PATH, directory)
            elif descriptors_parameter_file is None and var == "DESCRIPTORS_FOLDER":
                descriptors_parameter_file = os.path.join(ROOT_PATH, directory, desc_name + "_parameters.txt")
            elif var == "KEYPOINTS_EXECUTABLE":
                keypoint_executable = os.path.join(ROOT_PATH, directory)
            line = f.readline()

    if results_path is None or image_all_db_path is None or \
            descriptors_parameter_file is None or keypoint_executable is None:
        print("Error: could not find all of the necessary directory paths from the following file:")
        print(" ", os.path.join(ROOT_PATH, "project_structure.txt"))
        sys.exit(1)

    results_path = os.path.join(results_path, desc_name)

    image_db_path = os.path.join(image_all_db_path, database)
    if os.path.isdir(image_db_path):
        generate_keypoints(detector, desc_name, database)
