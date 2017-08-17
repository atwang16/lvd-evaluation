import subprocess
import sys
import os.path
import os

MIN_NUM_ARGS = 3
ROOT_PATH = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
PROJECT_TREE = os.path.join(ROOT_PATH, "project_structure.txt")
directories = {"results_path": None, "imageretrieval_executable": None, "fishervectors_executable": None}
flags = {"-generate_fishervectors": False, "-results": False}
file_output = None
PARAMETERS_FILE = os.path.join(os.getcwd(), "parameters.txt")
FISHER_PARAMETERS_FILE = os.path.join(os.getcwd(), "fisher_parameters.txt")
IMG_EXTENSIONS = [".jpg", ".png", ".ppm", ".pgm"]

query_sample_size = 5
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


def generate_fishervectors(desc, db):
    global directories
    desc_database_path = os.path.join(directories["results_path"], desc, db)
    results_db_path = desc_database_path
    dictionary_path = os.path.join(directories["results_path"], desc, desc + "_visual_dictionary.csv")

    args = [directories["fishervectors_executable"], FISHER_PARAMETERS_FILE, desc_database_path, results_db_path, dictionary_path]
    print(desc_name + ": " + "extracting fisher vectors from " + db)
    print("***")
    subprocess.run(args)


def generate_results(sequence):
    global file_output
    global query_sample_size
    results_seq_path = os.path.join(results_db_path, sequence)
    expand_results = sorted(os.listdir(results_seq_path))

    fisher_vectors = []
    for file in expand_results:
        if is_fisher(file):
            fisher_vectors.append(file)
    fisher_vectors.sort()

    if len(fisher_vectors) == 0:
        print("Error: no fisher vectors found.")
        return

    if query_sample_size < len(fisher_vectors):
        query_fisher_vectors = fisher_vectors[0:query_sample_size]
    else:
        query_fisher_vectors = fisher_vectors

    if len(query_fisher_vectors) == 0:
        print("Error: no fisher vectors found in", results_seq_path)

    for query_fv in query_fisher_vectors:
        query_fv_path = os.path.join(results_seq_path, query_fv)

        args = [directories["imageretrieval_executable"], desc_name, query_fv_path, results_db_path, "0", "1", file_output]

        print(desc_name + ": " + database + " - query " + query_fv)
        sys.stdout.flush()
        subprocess.run(args)


# MAIN #

if __name__ == "__main__":
    if len(sys.argv) < MIN_NUM_ARGS:
        print("Usage: python imageretrieval2.py desc_name database [-generate_fishervectors] [-results_only]")
        sys.exit(1)

    desc_name = sys.argv[1]
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

    results_db_path = os.path.join(directories["results_path"], desc_name, database)

    arg_index = MIN_NUM_ARGS
    while arg_index < len(sys.argv):
        for f in flags:
            if sys.argv[arg_index] == f:
                flags[f] = True
        arg_index += 1

    with open(PARAMETERS_FILE) as f:
        line = f.readline()
        while line != "":
            line_split = line.split("=")
            if len(line_split) >= 2:
                var = line_split[0]
                value = line_split[1]
                if value[-1] == "\n":
                    value = value[:-1]

                if var == "QUERY_SAMPLE_SIZE":
                    query_sample_size = int(value)
            line = f.readline()

    if flags["-generate_fishervectors"]:
        generate_fishervectors(desc_name, database)

    file_output = os.path.join(results_db_path, desc_name + "_" + database[0:3] + "_imageretrieval2.csv")
    result_sequences = sorted(os.listdir(results_db_path))

    if not flags["-results"]:
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