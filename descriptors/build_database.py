import subprocess
import sys
import os.path
import os

ROOT_PATH = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
results_path = None
image_all_db_path = None
image_db_path = None
descriptors_parameter_file = None


def generate_descriptors(desc, db):
    global descriptor_executable, descriptors_parameter_file, image_db_path, results_path
    args = [descriptor_executable, descriptors_parameter_file, image_db_path, results_path]
    print(desc + ": " + "extracting descriptors from " + db)
    print("***")
    subprocess.run(args)


# MAIN #

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python basetest.py desc_name [database_name]")
        sys.exit(1)

    desc_name = sys.argv[1]

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
                image_all_db_path = os.path.join(ROOT_PATH, directory)
            elif var == "DESCRIPTORS_FOLDER":
                descriptors_parameter_file = os.path.join(ROOT_PATH, directory, desc_name + "_parameters.txt")
            elif var == desc_name.upper() + "_EXECUTABLE":
                descriptor_executable = os.path.join(ROOT_PATH, directory)
            line = f.readline()

    if results_path is None or image_all_db_path is None or \
        descriptors_parameter_file is None or descriptor_executable is None:
        print("Error: could not find all of the necessary directory paths from the following file:")
        print(" ", os.path.join(ROOT_PATH, "project_structure.txt"))
        sys.exit(1)

    if len(sys.argv) >= 3:
        database = sys.argv[2]
        image_db_path = os.path.join(image_all_db_path, database)
        generate_descriptors(desc_name, database)
    else:
        all_databases = sorted(os.listdir(image_all_db_path))

        for db in all_databases:
            image_db_path = os.path.join(image_all_db_path, db)
            if os.path.isdir(image_db_path):
                generate_descriptors(desc_name, db)