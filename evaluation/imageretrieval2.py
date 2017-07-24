import subprocess
import sys
import os.path
import os
from random import sample
from shutil import copyfile, rmtree

ROOT_PATH = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
results_path = None
image_db_path = None
apptest2_executable = None
descriptor_executable = None
fisher_vector_executable = None
file_output = None
PARAMETERS_FILE = os.path.join(os.getcwd(), "parameters.txt")
FISHER_PARAMETERS_FILE = os.path.join(os.getcwd(), "fisher_parameters.txt")
IMG_EXTENSIONS = [".jpg", ".png", ".ppm", ".pgm"]

subset_seq_size = 30
query_sample_size = 5
generate_subset = False
generate_descs = False
generate_fishers = False
mean_ave_prec = 0.0
success_rate = 0.0
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
    global descriptor_executable, image_db_path, results_path

    args = [descriptor_executable, image_db_path, results_path]
    print(desc + ": " + "extracting descriptors from " + db)
    print("***")
    subprocess.run(args)


def generate_fisher_vectors(desc, db):
    global fisher_vector_executable, results_path
    desc_database_path = os.path.join(results_path, desc, db)
    results_db_path = desc_database_path
    dictionary_path = os.path.join(results_path, desc, desc + "_visual_dictionary.csv")

    args = [fisher_vector_executable, FISHER_PARAMETERS_FILE, desc_database_path, results_db_path, dictionary_path]
    print(desc_name + ": " + "extracting fisher vectors from " + db)
    print("***")
    subprocess.run(args)


def generate_results(sequence):
    global apptest2_executable, file_output
    global query_sample_size
    results_seq_path = os.path.join(results_db_path, sequence)
    expand_results = sorted(os.listdir(results_seq_path))

    fisher_vectors = []
    for file in expand_results:
        if is_fisher(file):
            fisher_vectors.append(file)
    sorted(fisher_vectors)

    if query_sample_size < len(fisher_vectors):
        query_fisher_vectors = sample(fisher_vectors, query_sample_size)
    else:
        query_fisher_vectors = fisher_vectors

    if len(query_fisher_vectors) == 0:
        print("Error: no fisher vectors found in", results_seq_path)

    for query_fv in query_fisher_vectors:
        query_fv_path = os.path.join(results_seq_path, query_fv)

        args = [apptest2_executable, desc_name, query_fv_path, results_db_path, "0", file_output]

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

    with open(os.path.join(ROOT_PATH, "project_structure.txt")) as f:
        line = f.readline()
        while line != "":
            line_split = line.split("=")
            var = line_split[0]
            directory = line_split[-1]
            if(directory[-1] == "\n"):
                directory = directory[:-1]
            if var == "RESULTS_FOLDER":
                results_path = os.path.join(ROOT_PATH, directory)
            elif var == "DATASETS_FOLDER":
                image_db_path = os.path.join(ROOT_PATH, directory, database)
            elif var == "APPTEST2_EXECUTABLE":
                apptest2_executable = os.path.join(ROOT_PATH, directory)
            elif var == "FISHER_VECTOR_EXECUTABLE":
                fisher_vector_executable = os.path.join(ROOT_PATH, directory)
            elif var == desc_name.upper() + "_EXECUTABLE":
                descriptor_executable = os.path.join(ROOT_PATH, directory)
            line = f.readline()

    if results_path is None or image_db_path is None or apptest2_executable is None or \
                    fisher_vector_executable is None or descriptor_executable is None:
        print("Error: could not find all of the necessary directory paths from the following file:")
        print(" ", os.path.join(ROOT_PATH, "project_structure.txt"))
        sys.exit(1)

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
    results_db_path = os.path.join(results_path, desc_name, database)

    with open(PARAMETERS_FILE) as f:
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
                for f in seq_files:
                    if is_image(f):
                        seq_images.append(f)

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
    if generate_fishers:
        generate_fisher_vectors(desc_name, database)

    file_output = os.path.join(results_db_path, desc_name + "_" + database[0:3] + "_imageretrieval2.csv")
    result_sequences = sorted(os.listdir(results_db_path))

    for r_seq in result_sequences:
        if os.path.isdir(os.path.join(results_db_path, r_seq)) and r_seq != "clutter":
            generate_results(r_seq)

    if os.path.exists(file_output):
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

    if num_queries > 0:
        mean_ave_prec /= num_queries
        success_rate /= num_queries

        print("Mean Average Precision:", mean_ave_prec)
        print("Success Rate:", success_rate)
    else:
        print("No queries found.")