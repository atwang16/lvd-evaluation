import subprocess
import sys
import os.path
import os

ROOT_PATH = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
RESULTS_PATH = os.path.join(ROOT_PATH, "results")
IMG_DB_PATH = os.path.join(ROOT_PATH, "datasets")

def generate_descriptors(desc, db):
    img_database_path = os.path.join(IMG_DB_PATH, db)
    executable_path = os.path.join(ROOT_PATH, "build_ds_" + desc, desc)

    args = [executable_path, img_database_path + os.sep, res_path + os.sep]
    print(desc_name + ": " + "extracting descriptors from " + db)
    print("-------------")
    subprocess.run(args)

####### MAIN #######

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python basetest.py desc_name [database_name]")
        sys.exit(1)

    desc_name = sys.argv[1]
    res_path = os.path.join(RESULTS_PATH, desc_name)

    if len(sys.argv) >= 3:
        database = sys.argv[2]
        generate_descriptors(desc_name, database)
    else:
        all_databases = os.listdir(IMG_DB_PATH)

        for db in all_databases:
            if os.path.isdir(os.path.join(IMG_DB_PATH, db)):
                generate_descriptors(desc_name, db)