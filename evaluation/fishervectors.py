import subprocess
import sys
import os.path
import os

ROOT_PATH = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
results_path = None
fisher_vector_executable = None
PARAMETERS_PATH = os.path.join(os.getcwd(), "fisher_parameters.txt")


def generate_fisher_vectors(desc_name, db):
    global fisher_vector_executable, results_path
    desc_database_path = os.path.join(results_path, desc_name, db)
    dictionary_path = os.path.join(results_path, desc_name, desc_name + "_visual_dictionary.csv")

    if not os.path.isdir(results_path):
        os.makedirs(results_path)

    args = [fisher_vector_executable, PARAMETERS_PATH, desc_database_path, desc_database_path, dictionary_path]
    print(desc_name + ": " + "extracting fisher vectors from " + db)
    print("***")
    subprocess.run(args)


# MAIN #

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python fishervectors.py desc_name [database_name]")
        sys.exit(1)

    desc_name = sys.argv[1]

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
            elif var == "FISHER_VECTOR_EXECUTABLE":
                fisher_vector_executable = os.path.join(ROOT_PATH, directory)
            line = f.readline()

    if results_path is None or fisher_vector_executable is None:
        print("Error: could not find all of the necessary directory paths from the following file:")
        print(" ", os.path.join(ROOT_PATH, "project_structure.txt"))
        sys.exit(1)

    desc_results_path = os.path.join(results_path, desc_name)

    if not os.path.isdir(desc_results_path):
        os.makedirs(desc_results_path)

    # Generate fisher vectors for a single database
    if len(sys.argv) >= 3:
        database = sys.argv[2]
        generate_fisher_vectors(desc_name, database)
    # Generate fisher vectors for all databases
    else:
        all_databases = os.listdir(desc_results_path)

        for db in all_databases:
            if os.path.isdir(os.path.join(desc_results_path, db)):
                generate_fisher_vectors(desc_name, db)