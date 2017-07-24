import subprocess
import sys
import os.path
import os
from random import sample
from shutil import copyfile, rmtree

ROOT_PATH = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
results_path = None
image_db_path = None
apptest_executable = None
descriptor_parameters_path = None
PARAMETERS_FILE_PATH = os.path.join(os.getcwd(), "parameters.txt")

IMG_EXTENSIONS = [".jpg", ".png", ".ppm", ".pgm"]
distance = "L2"
subset_seq_size = 30
query_sample_size = 5
dist_threshold = float('inf')
generate_subset = False
generate_descs = False
mean_ave_prec = 0
success_rate = 0
num_queries = 0


def is_image(fname):
    _, ext = os.path.splitext(fname)
    return ext in IMG_EXTENSIONS


def is_desc(fname):
    fname_split = fname.split("_")
    return fname_split[-1] == "ds.csv"


def generate_descriptors(desc, db):
    global descriptor_executable, image_db_path, results_path

    args = [descriptor_executable, image_db_path, results_path]
    print(desc + ": " + "extracting descriptors from " + db)
    print("***")
    subprocess.run(args)


def generate_results(sf):
    global results_db_path, apptest_executable
    global mean_ave_prec, success_rate, num_queries, query_sample_size, file_output
    results_sf_path = os.path.join(results_db_path, sf)
    expand_results_sf = sorted(os.listdir(results_sf_path))

    descriptors = []
    for file in expand_results_sf:
        if is_desc(file):
            descriptors.append(file)
    sorted(descriptors)

    if query_sample_size < len(descriptors):
        query_descriptors = sample(descriptors, query_sample_size)
    else:
        query_descriptors = descriptors

    for query_ds in query_descriptors:
        query_ds_path = os.path.join(results_sf_path, query_ds)

        args = [apptest_executable, PARAMETERS_FILE_PATH, desc_name, str(dist_threshold), distance, query_ds_path,
                results_db_path, "0", file_output]

        print(desc_name + ": " + database + " - query " + query_ds)
        subprocess.run(args)


# MAIN #

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python imageretrieval.py desc_name database [-generate_subset] [-generate_descriptors]")
        sys.exit(1)

    desc_name = sys.argv[1]
    database = sys.argv[2]

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
                image_db_path = os.path.join(ROOT_PATH, directory, database)
            elif var == "APPTEST_EXECUTABLE":
                apptest_executable = os.path.join(ROOT_PATH, directory)
            elif var == "DESCRIPTORS_FOLDER":
                descriptor_parameters_path = os.path.join(ROOT_PATH, directory, desc_name + "_parameters.txt")
            elif var == desc_name.upper() + "_EXECUTABLE":
                descriptor_executable = os.path.join(ROOT_PATH, directory)
            line = f.readline()

    if results_path is None or image_db_path is None or apptest_executable is None or \
                    descriptor_parameters_path is None or descriptor_executable is None:
        print("Error: could not find all of the necessary directory paths from the following file:")
        print(" ", os.path.join(ROOT_PATH, "project_structure.txt"))
        sys.exit(1)

    arg_index = 3
    while arg_index < len(sys.argv):
        if sys.argv[arg_index] == "-generate_subset":
            generate_subset = True
            generate_descs = True
        elif sys.argv[arg_index] == "-generate_descriptors":
            generate_descs = True
        arg_index += 1

    results_db_path = os.path.join(results_path, desc_name, database)

    # Get distance metric and distance threshold from file
    with open(descriptor_parameters_path) as f:
        line = f.readline()
        while line != "":
            line_split = line.split("=")
            var = line_split[0]
            value = line_split[-1]
            if value[-1] == "\n":
                value = value[:-1]

            if var == "distance":
                distance = value
            elif var == "distancethreshold":
                dist_threshold = value
            line = f.readline()

    with open(PARAMETERS_FILE_PATH) as f:
        line = f.readline()
        while line != "":
            line_split = line.split("=")
            var = line_split[0]
            value = line_split[-1]
            if value[-1] == "\n":
                value = value[:-1]

            if var == "SEQUENCE_SIZE":
                subset_seq_size = int(value)
            elif var == "QUERY_SAMPLE_SIZE":
                query_sample_size = int(value)
            line = f.readline()

            # generate subset of desired database
        if generate_subset:
            database += "_subset"
            new_img_db_path = os.path.join(image_db_path, os.pardir(), database)
            if os.path.isdir(new_img_db_path):
                rmtree(new_img_db_path)

            os.mkdir(new_img_db_path)
            db_sequences = sorted(os.listdir(image_db_path))

            for seq in db_sequences:
                old_seq = os.path.join(image_db_path, seq)
                new_seq = os.path.join(new_img_db_path, seq)

                if os.path.isdir(old_seq):
                    os.mkdir(new_seq)
                    seq_files = sorted(os.listdir(old_seq))
                    seq_images = []

                    # remove non-images
                    for file in seq_files:
                        if is_image(file):
                            seq_images.append(file)

                    # find subset of images
                    if subset_seq_size < len(seq_images):
                        subset_images = sample(seq_images, subset_seq_size)
                    else:
                        subset_images = seq_images

                    # copy images to new directory
                    for img in subset_images:
                        copyfile(os.path.join(old_seq, img), os.path.join(new_seq, img))

            image_db_path = new_img_db_path
            results_db_path = os.path.join(results_path, desc_name, database)
        if generate_descs:
            generate_descriptors(desc_name, database)

    file_output = os.path.join(results_db_path, desc_name + "_" + database[0:3] + "_imageretrieval.csv")
    result_sequences = sorted(os.listdir(results_db_path))

    for r_seq in result_sequences:
        if os.path.isdir(os.path.join(results_db_path, r_seq)) and r_seq != "clutter":
            generate_results(r_seq)

    with open(file_output) as f:
        line = f.readline()
        while line != "":
            line_split = line.split(",")
            ave_precision = float(line_split[1])
            success = int(line_split[2])
            mean_ave_prec += ave_precision
            success_rate += success
            num_queries += 1

            line = f.readline()

    mean_ave_prec /= num_queries
    success_rate /= num_queries

    print("Mean Average Precision:", mean_ave_prec)
    print("Success Rate:", success_rate)
