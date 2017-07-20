import subprocess
import sys
import os.path
import os

ROOT_PATH = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
results_path = None
results_db_path = None
image_db_path =  None
basetest_executable = None
descriptor_parameters_path = None
PARAMETERS_FILE_PATH = os.path.join(os.getcwd(), "parameters.txt")
IMG_EXTENSIONS = [".jpg", ".png", ".ppm", ".pgm"]
distance = "L2"
DESCRIPTOR_SUFFIX = "ds.csv"
KEYPOINT_SUFFIX = "kp.csv"
generate_descs = False

def is_image(fname):
    _, ext = os.path.splitext(fname)
    return ext in IMG_EXTENSIONS


def generate_descriptors(desc, db):
    global descriptor_executable, descriptors_parameter_file, image_db_path, results_path
    args = [descriptor_executable, descriptors_parameter_file, image_db_path, results_path]
    print(desc + ": " + "extracting descriptors from " + db)
    print("***")
    subprocess.run(args)


def generate_results(desc, sequence):
    global basetest_executable, image_db_path, results_db_path
    img_num = 2
    image_seq_path = os.path.join(image_db_path, sequence)
    results_seq_path = os.path.join(results_db_path, sequence)
    image_seq_expanded = os.listdir(image_seq_path)

    images = []
    for file in image_seq_expanded:
        if is_image(file):
            images.append(file)
    sorted(images)

    img1_path = os.path.join(image_seq_path, images[0])
    img1_base_name, _ = os.path.splitext(images[0])
    img_seq = images[0].split("_")[0] + "_" + images[0].split("_")[1]
    desc1_path = os.path.join(results_seq_path, img1_base_name + "_" + DESCRIPTOR_SUFFIX)
    kp1_path = os.path.join(results_seq_path, img1_base_name + "_" + KEYPOINT_SUFFIX)

    while img_num <= len(images):
        current_img = images[img_num - 1]
        img_base_name, _ = os.path.splitext(current_img)
        current_ds = img_base_name + "_" + DESCRIPTOR_SUFFIX
        current_kp = img_base_name + "_" + KEYPOINT_SUFFIX
        current_hom = "H1to" + str(img_num) + "p"

        img2_path = os.path.join(image_seq_path, current_img)
        desc2_path = os.path.join(results_seq_path, current_ds)
        kp2_path = os.path.join(results_seq_path, current_kp)
        hom_path = os.path.join(image_seq_path, current_hom)
        stat_results_file = os.path.join(results_seq_path, img_seq + "_stat_001" + "_" + '{0:03d}'.format(img_num) + ".txt")
        draw_results_file = os.path.join(results_seq_path, img_seq + "_draw_001" + "_" + '{0:03d}'.format(img_num) + ".png")

        args = [basetest_executable, PARAMETERS_FILE_PATH, desc, img1_path, desc1_path, kp1_path, img2_path, desc2_path,
                kp2_path, hom_path, distance, stat_results_file, draw_results_file]

        print(desc_name + ": " + database + " - " + sequence + ": " + images[0] + " to " + current_img)
        subprocess.run(args)
        img_num += 1

# MAIN #

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python basetest.py desc_name database_name [-generate_descriptors]")
        sys.exit(1)

    desc_name = sys.argv[1]
    database = sys.argv[2]

    arg_index = 3
    while arg_index < len(sys.argv):
        if sys.argv[arg_index] == "-generate_descriptors":
            generate_descs = True
        arg_index += 1

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
            elif var == "BASETEST_EXECUTABLE":
                basetest_executable = os.path.join(ROOT_PATH, directory)
            elif var == "DESCRIPTORS_FOLDER":
                descriptor_parameters_path = os.path.join(ROOT_PATH, directory, desc_name + "_parameters.txt")
            elif var == desc_name.upper() + "_EXECUTABLE":
                descriptor_executable = os.path.join(ROOT_PATH, directory)
            line = f.readline()

    if results_path is None or image_db_path is None or basetest_executable is None or descriptor_parameters_path is None \
            or descriptor_executable is None:
        print("Error: could not find all of the necessary directory paths from the following file:")
        print(" ", os.path.join(ROOT_PATH, "project_structure.txt"))
        sys.exit(1)

    if generate_descs:
        generate_descriptors(desc_name, database)

    results_db_path = os.path.join(results_path, desc_name, database)

    with open(descriptor_parameters_path) as f:
        line = f.readline()
        while line != "":
            line_split = line.split("=")
            var = line_split[0]
            value = line_split[-1]
            if(value[-1] == "\n"):
                value = value[:-1]
            if var == "distance":
                distance = value
            line = f.readline()

    db_sequences = os.listdir(results_db_path)

    for seq in db_sequences:
        if os.path.isdir(os.path.join(results_db_path, seq)):
            generate_results(desc_name, seq)