import subprocess
import sys
import os

MIN_NUM_ARGS = 3
ROOT_PATH = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
PROJECT_TREE = os.path.join(ROOT_PATH, "project_structure.txt")
directories = {"results_path": None, "datasets_path": None, "descriptors_path": None, "keypoints_executable": None}
flags = {"-overwrite": "0"}
desc_name = None


def generate_keypoints(det, db):
    global descriptors_parameter_file, directories, flags
    args = [directories["keypoint_executable"], det, descriptors_parameter_file, directories["image_db_path"], directories["results_path"], flags["-overwrite"]]
    print(det + ": " + "Extracting keypoints from " + db)
    print("***")
    subprocess.run(args)


# MAIN #

if __name__ == "__main__":
    if len(sys.argv) < MIN_NUM_ARGS:
        print("Usage: python3 build_keypoints.py detector database_name [descriptor_name] [-overwrite]")
        sys.exit(1)

    detector = sys.argv[1]
    database = sys.argv[2]

    if len(sys.argv) > MIN_NUM_ARGS and sys.argv[MIN_NUM_ARGS][0] != "-":
        desc_name = sys.argv[3]
    else:
        descriptors_parameter_file = "null"

    with open(PROJECT_TREE) as f:
        line = f.readline()
        while line != "":
            line_split = line.split("=")
            if len(line_split) >= 2:
                var = line_split[0].lower()
                directory = line_split[1].strip()
                if var in directories:
                    directories[var] = os.path.join(ROOT_PATH, directory)
            line = f.readline()

    for d in directories:
        if directories[d] is None:
            print("Error: could not find", directories[d].upper(), "in file:")
            print(" ", PROJECT_TREE)
            sys.exit(1)

    arg_index = MIN_NUM_ARGS
    while arg_index < len(sys.argv):
        for f in flags:
            if sys.argv[arg_index] == f:
                flags[f] = "1"
        arg_index += 1

    if descriptors_parameter_file != "null":
        descriptors_parameter_file = os.path.join(directories["descriptors_path"], desc_name + "_parameters.txt")
    directories["results_path"] = os.path.join(directories["results_path"], detector)
    directories["datasets_path"] = os.path.join(directories["datasets_path"], database)

    generate_keypoints(detector, database)
