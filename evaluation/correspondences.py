import subprocess
import sys
import os.path
import os

MIN_NUM_ARGS = 3
ROOT_PATH = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
PROJECT_TREE = os.path.join(ROOT_PATH, "project_structure.txt")
directories = {"results_path": None, "datasets_path": None, "correspondences_executable": None}
results_kp_path = None
file_output = None
PARAMETERS_FILE_PATH = os.path.join(os.getcwd(), "basetest_parameters.txt")
KEYPOINT_SUFFIX = "kp.csv"
kp_dist_thresh = 2.5


def is_keypoint(file):
    return file.split("_")[-1] == KEYPOINT_SUFFIX


def generate_results(det, sequence):
    global directories
    img_num = 2
    image_seq_path = os.path.join(directories["datasets_path"], sequence)
    kp_seq_path = os.path.join(directories["results_path"], sequence)
    kp_seq_expanded = os.listdir(kp_seq_path)

    # print(kp_seq_expanded)

    keypoints = []
    for file in kp_seq_expanded:
        if is_keypoint(file):
            keypoints.append(file)
    keypoints.sort()

    name_split = keypoints[0].split("_")
    base_name = name_split[0] + "_" + name_split[1]
    kp1_path = os.path.join(kp_seq_path, base_name + "_" + '{0:03d}'.format(1) + "_kp.csv")

    if os.path.exists(kp1_path):
        while img_num <= len(keypoints):
            current_kp = base_name + "_" + '{0:03d}'.format(img_num) + "_kp.csv"
            kp2_path = os.path.join(kp_seq_path, current_kp)
            hom_path = os.path.join(image_seq_path, "H1to" + str(img_num) + "p")
            results_file = os.path.join(kp_seq_path, base_name + "_" + '{0:03d}'.format(img_num) + "_co.csv")

            if os.path.exists(kp2_path) and os.path.exists(hom_path):
                args = [directories["correspondences_executable"], kp1_path, kp2_path, hom_path, kp_dist_thresh, results_file]
                print(det + ": " + database + " - " + sequence + ": " + keypoints[0] + " to " + current_kp)
                subprocess.run(args)
            img_num += 1

# MAIN #

if __name__ == "__main__":
    if len(sys.argv) < MIN_NUM_ARGS:
        print("Usage: python3 correspondences.py detector_name database_name")
        sys.exit(1)

    detector = sys.argv[1]
    database = sys.argv[2]

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

    directories["datasets_path"] = os.path.join(directories["datasets_path"], database)
    directories["results_path"] = os.path.join(directories["results_path"], detector, database)

    with open(PARAMETERS_FILE_PATH) as f:
        line = f.readline()
        while line != "":
            line_split = line.split("=")
            if len(line_split) >= 2:
                var = line_split[0]
                value = line_split[1]
                if value[-1] == "\n":
                    value = value[:-1]
                if var == "KP_DIST_THRESH":
                    kp_dist_thresh = value
            line = f.readline()

    db_sequences = sorted(os.listdir(directories["results_path"]))

    for seq in db_sequences:
        if os.path.isdir(os.path.join(directories["results_path"], seq)):
            generate_results(detector, seq)
