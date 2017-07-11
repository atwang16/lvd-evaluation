import subprocess
import sys
import os.path
import os

root_folder = os.path.join(os.getcwd(), "..")
results_folder = "results"
image_folder = "datasets"
results_dir = os.path.join(root_folder, results_folder)
image_dir = os.path.join(root_folder, image_folder)
BASE_TEST_PATH_DIR = os.path.join(root_folder, "basetest", "basetest")
PARAMETERS_FILE_PATH = os.path.join(root_folder, "evaluation", "parameters.txt")
DESCRIPTOR_PARAMETERS_DIR = os.path.join(root_folder, "descriptors")
distance = "L2"
dist_threshold = float('inf')
calc_descriptors = True

def generate_descriptors(image_dir):
    pass


####### MAIN #######

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python imageretrieval.py desc_name database generate_descriptors")
        sys.exit(1)

    desc_name = sys.argv[1]
    database = sys.argv[2]
    calc_descriptors = (int(sys.argv[3]) == 1)
    image_dir = os.path.join(image_dir, database)
    results_dir = os.path.join(results_dir, desc_name, database)

    # Get distance metric and distance threshold from file
    with open(os.path.join(DESCRIPTOR_PARAMETERS_DIR, desc_name + "_parameters.txt")) as f:
        line = f.readline()
        while line != "":
            line_split = line.split("=")
            var = line_split[0]
            value = line_split[-1]
            if(value[-1] == "\n"):
                value = value[:-1]

            if var == "distance":
                distance = value
            elif var == "distancethreshold":
                dist_threshold = value
            line = f.readline()

    # generate threshold values
    subfolders = os.listdir(results_dir)

    # calculate descriptors and keypoints for each database image
    if calc_descriptors:
        generate_descriptors(image_dir)