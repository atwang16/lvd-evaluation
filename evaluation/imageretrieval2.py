import subprocess
import sys
import os.path
import os
from random import sample
from shutil import copyfile

ROOT_PATH = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
RESULTS_PATH = os.path.join(ROOT_PATH, "results")
IMG_DB_PATH = os.path.join(ROOT_PATH, "datasets")
APP_TEST_BUILD = os.path.join(ROOT_PATH, "build_apptest2", "imageretrieval2")
PARAMETERS_FILE_PATH = os.path.join(ROOT_PATH, "evaluation", "parameters.txt")
FISHER_PARAMETERS_FILE = os.path.join(ROOT_PATH, "evaluation", "fisher_parameters.txt")
FISHER_VECTOR_EXECUTABLE = os.path.join(ROOT_PATH, "build_fisher", "fishervectors")
IMG_EXTENSIONS = [".jpg", ".png", ".ppm", ".pgm"]
seq_size = 30
query_sample_size = 5
generate_subset = False
generate_descs = False
generate_fishers = False
mean_ave_prec = 0
success_rate = 0
num_queries = 0


def is_image(fname):
    _, ext = os.path.splitext(fname)
    return ext in IMG_EXTENSIONS


def is_desc(fname):
    fname_split = fname.split("_")
    return fname_split[-1] == "ds.csv"


def is_fisher(fname):
    fname_split = fname.split("_")
    return fname_split[-1] == "fv.csv"


def generate_descriptors(desc, db):
    img_database_path = os.path.join(IMG_DB_PATH, db)
    executable_path = os.path.join(ROOT_PATH, "build_ds_" + desc, desc)

    args = [executable_path, img_database_path + os.sep, RESULTS_PATH + os.sep]
    print(desc_name + ": " + "extracting descriptors from " + db)
    print("***")
    subprocess.run(args)


def generate_fisher_vectors(desc, db):
    desc_database_path = os.path.join(RESULTS_PATH, desc, db)
    results_path = os.path.join(RESULTS_PATH, desc, db)
    dictionary_path = os.path.join(RESULTS_PATH, desc, desc + "_visual_dictionary.csv")

    args = [FISHER_VECTOR_EXECUTABLE, FISHER_PARAMETERS_FILE, desc_database_path + os.sep,
            results_path + os.sep, dictionary_path]
    print(desc_name + ": " + "extracting fisher vectors from " + db)
    print("***")
    subprocess.run(args)


def generate_results(subfolder):
    global mean_ave_prec, success_rate, num_queries, query_sample_size, file_output
    results_sf_path = os.path.join(results_path, subfolder)
    expand_results_sf = os.listdir(results_sf_path)

    fisher_vectors = []
    for file in expand_results_sf:
        if is_fisher(file):
            fisher_vectors.append(file)
    sorted(fisher_vectors)

    if query_sample_size < len(fisher_vectors):
        query_fisher_vectors = sample(fisher_vectors, query_sample_size)
    else:
        query_fisher_vectors = fisher_vectors

    for query_fv in query_fisher_vectors:
        query_fv_path = os.path.join(results_sf_path, query_fv)

        args = [APP_TEST_BUILD, desc_name, query_fv_path, results_path, "0", file_output]

        print(desc_name + ": " + database + " - query " + query_fv)
        sys.stdout.flush()
        subprocess.run(args)


# MAIN #

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python imageretrieval2.py desc_name database [-generate_subset] [-generate_descriptors] "
              "[-generate_fishervectors]")
        sys.exit(1)

    desc_name = sys.argv[1]
    database = sys.argv[2]
    arg_index = 3
    while arg_index < len(sys.argv):
        if sys.argv[arg_index] == "-generate_subset":
            generate_subset = True
            generate_descs = True
            generate_fishers = True
        elif sys.argv[arg_index] == "-generate_descriptors":
            generate_descs = True
            generate_fishers = True
        elif sys.argv[arg_index] == "-generate_fishervectors":
            generate_fishers = True
        arg_index += 1
    image_path = os.path.join(IMG_DB_PATH, database)
    results_path = os.path.join(RESULTS_PATH, desc_name, database)

    with open(PARAMETERS_FILE_PATH) as f:
        line = f.readline()
        while line != "":
            line_split = line.split("=")
            var = line_split[0]
            value = line_split[-1]
            if value[-1] == "\n":
                value = value[:-1]

            if var == "SEQUENCE_SIZE":
                seq_size = int(value)
            elif var == "QUERY_SAMPLE_SIZE":
                query_sample_size = int(value)
            line = f.readline()

    # generate subset of desired database
    if generate_subset:
        database += "_subset"
        new_db_path = os.path.join(IMG_DB_PATH, database)
        if not os.path.isdir(new_db_path):
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

        image_path = new_db_path
        results_path = os.path.join(RESULTS_PATH, desc_name, database)
    if generate_descs:
        generate_descriptors(desc_name, database)
    if generate_fishers:
        generate_fisher_vectors(desc_name, database)

    file_output = os.path.join(results_path, desc_name + "_" + database[0:3] + "_imageretrieval2.csv")
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
            mean_ave_prec += ave_precision
            success_rate += success
            num_queries += 1

            line = f.readline()

        mean_ave_prec /= num_queries
    success_rate /= num_queries

    print("Mean Average Precision:", mean_ave_prec)
    print("Success Rate:", success_rate)