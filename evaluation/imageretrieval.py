import subprocess
import sys
import os.path
import os
from random import sample
from shutil import copyfile

ROOT_PATH = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
RESULTS_PATH = os.path.join(ROOT_PATH, "results")
IMG_DB_PATH = os.path.join(ROOT_PATH, "datasets")
APP_TEST_BUILD = os.path.join(ROOT_PATH, "build_apptest", "imageretrieval")
PARAMETERS_FILE_PATH = os.path.join(ROOT_PATH, "evaluation", "parameters.txt")
DESCRIPTOR_PARAMETERS_DIR = os.path.join(ROOT_PATH, "descriptors")
IMG_EXTENSIONS = [".jpg", ".png", ".ppm", ".pgm"]
distance = "L2"
seq_size = 30
query_sample_size = 5
dist_threshold = float('inf')
generate_subset = False
map = 0
success_rate = 0
num_queries = 0

def is_image(fname):
    _, ext = os.path.splitext(fname)
    return ext in IMG_EXTENSIONS

def is_desc(fname):
    fname_split = fname.split("_")
    return fname_split[-1] == "ds.csv"

def generate_descriptors(desc, db):
    img_database_path = os.path.join(IMG_DB_PATH, db)
    executable_path = os.path.join(ROOT_PATH, "build_ds_" + desc, desc)

    args = [executable_path, img_database_path + os.sep, RESULTS_PATH + os.sep]
    print(desc_name + ": " + "extracting descriptors from " + db)
    print("-------------")
    subprocess.run(args)

def generate_results(sf):
    global map, success_rate, num_queries, query_sample_size, file_output
    results_sf_path = os.path.join(results_path, sf)
    expand_results_sf = os.listdir(results_sf_path)

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

        args = [APP_TEST_BUILD, PARAMETERS_FILE_PATH, desc_name, dist_threshold, distance, query_ds_path,
                results_path, "0", file_output]

        print(desc_name + ": " + database + " - query " + query_ds)
        sys.stdout.flush()

        # cp = subprocess.run(args, stdout=subprocess.PIPE, encoding="utf-8")
        # output = cp.stdout.split(" ")
        subprocess.run(args)

####### MAIN #######

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python imageretrieval.py desc_name database [-generate_subset]")
        sys.exit(1)

    desc_name = sys.argv[1]
    database = sys.argv[2]
    if len(sys.argv) >= 4:
        generate_subset = (sys.argv[3] == "-generate_subset")
    image_path = os.path.join(IMG_DB_PATH, database)
    results_path = os.path.join(RESULTS_PATH, desc_name, database)

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

    with open(PARAMETERS_FILE_PATH) as f:
        line = f.readline()
        while line != "":
            line_split = line.split("=")
            var = line_split[0]
            value = line_split[-1]
            if (value[-1] == "\n"):
                value = value[:-1]

            if var == "SEQUENCE_SIZE":
                seq_size = int(value)
            elif var == "QUERY_SAMPLE_SIZE":
                query_sample_size = int(value)
            line = f.readline()

    # generate subset of desired database
    if generate_subset:
        new_db = database + "_subset"
        new_db_path = os.path.join(IMG_DB_PATH, new_db)
        os.mkdir(new_db_path)

        seqs = os.listdir(image_path)

        for sq in seqs:
            old_sq = os.path.join(image_path, sq)
            new_sq = os.path.join(new_db_path, sq)
            if os.path.isdir(old_sq):
                os.mkdir(new_sq)
                seq_files = os.listdir(old_sq)
                seq_images = []

                # remove non-images
                for f in seq_files:
                    if is_image(f):
                        seq_images.append(f)

                # find subset of images
                if seq_size < len(seq_images):
                    subset_images = set(sample(seq_images, seq_size))
                else:
                    subset_images = seq_images

                for img in subset_images:
                    copyfile(os.path.join(old_sq, img), os.path.join(new_sq, img))

        generate_descriptors(desc_name, new_db)
        image_path = new_db_path
        results_path = os.path.join(RESULTS_PATH, desc_name, new_db)

    file_output = os.path.join(results_path, desc_name + "_" + database[0:3] + "_imageretrieval.csv")
    subfolders = os.listdir(results_path)

    for sf in subfolders:
        if os.path.isdir(os.path.join(results_path, sf)) and sf != "clutter":
            generate_results(sf)

    with open(file_output) as f:
        line = f.readline()
        while line != "":
            line_split = line.split(",")
            ave_precision = float(line_split[1])
            success = int(line_split[2])
            map += ave_precision
            success_rate += success
            num_queries += 1

            line = f.readline()

    map /= num_queries
    success_rate /= num_queries

    print("Mean Average Precision:", map)
    print("Success Rate:", success_rate)