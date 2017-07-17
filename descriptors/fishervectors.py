import subprocess
import sys
import os.path
import os

ROOT_PATH = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
RESULTS_PATH = os.path.join(ROOT_PATH, "results")
DESC_DB_PATH = os.path.join(ROOT_PATH, "results")
EXECUTABLE_PATH = os.path.join(ROOT_PATH, "build_fisher", "fishervectors")
PARAMETERS_PATH = os.path.join(ROOT_PATH, "descriptors", "fisher_parameters.txt")

def generate_fisher_vectors(desc_name, db):
    desc_database_path = os.path.join(DESC_DB_PATH, desc_name, db)

    args = [EXECUTABLE_PATH, PARAMETERS_PATH, desc_database_path + os.sep, RESULTS_PATH + os.sep]
    print(desc_name + ": " + "extracting fisher vectors from " + db)
    print("***")
    subprocess.run(args)

####### MAIN #######

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python fishervectors.py desc_name [database_name]")
        sys.exit(1)

    desc_name = sys.argv[1]

    if len(sys.argv) >= 3:
        database = sys.argv[2]
        generate_fisher_vectors(desc_name, database)
    else:
        all_databases = os.listdir(DESC_DB_PATH)

        for db in all_databases:
            if os.path.isdir(os.path.join(DESC_DB_PATH, db)):
                generate_fisher_vectors(desc_name, db)