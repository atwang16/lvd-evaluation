import subprocess
import sys
import os.path
import os

MIN_NUM_ARGS = 3
ROOT_PATH = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
PROJECT_TREE = os.path.join(ROOT_PATH, "project_structure.txt")
directories = {"results_path": None, "fishervectors_executable": None}
results_path = None
fisher_vector_executable = None
PARAMETERS_PATH = os.path.join(os.getcwd(), "fisher_parameters.txt")


def generate_fisher_vectors(desc_name, db):
    global directories
    desc_database_path = os.path.join(directories["results_path"], desc_name, db)
    dictionary_path = os.path.join(directories["results_path"], desc_name, desc_name + "_visual_dictionary.csv")

    args = [directories["fishervectors_executable"], PARAMETERS_PATH, desc_database_path, desc_database_path, dictionary_path]
    print(desc_name + ": " + "extracting fisher vectors from " + db)
    print("***")
    subprocess.run(args)


# MAIN #

if __name__ == "__main__":
    if len(sys.argv) < MIN_NUM_ARGS:
        print("Usage: python fishervectors.py desc_name database_name")
        sys.exit(1)

    descriptor = sys.argv[1]
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

    generate_fisher_vectors(descriptor, database)
