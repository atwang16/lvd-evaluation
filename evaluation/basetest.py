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
file_output = None
PARAMETERS_FILE_PATH = os.path.join(os.getcwd(), "parameters.txt")
IMG_EXTENSIONS = [".jpg", ".png", ".ppm", ".pgm"]
distance = "L2"
DESCRIPTOR_SUFFIX = "ds.csv"
KEYPOINT_SUFFIX = "kp.csv"
generate_descs = False
results_only = False
MATCH_RATIO = 0
MATCHING_SCORE = 1
PRECISION = 2
RECALL = 3
MATCH_TIME = 4
NUM_QUERIES = 5

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
    global basetest_executable, image_db_path, results_db_path, file_output
    img_num = 2
    image_seq_path = os.path.join(image_db_path, sequence)
    results_seq_path = os.path.join(results_db_path, sequence)
    image_seq_expanded = sorted(os.listdir(image_seq_path))

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
        draw_results_file = os.path.join(results_seq_path, img_seq + "_draw_001" + "_" + '{0:03d}'.format(img_num) + ".png")

        args = [basetest_executable, PARAMETERS_FILE_PATH, desc, img1_path, desc1_path, kp1_path, img2_path, desc2_path,
                kp2_path, hom_path, distance, "-s", file_output, "-d", draw_results_file]

        print(desc_name + ": " + database + " - " + sequence + ": " + images[0] + " to " + current_img)
        subprocess.run(args)
        img_num += 1

# MAIN #

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python basetest.py desc_name database_name [-generate_descriptors] [-results_only]")
        sys.exit(1)

    desc_name = sys.argv[1]
    database = sys.argv[2]

    arg_index = 3
    while arg_index < len(sys.argv):
        if sys.argv[arg_index] == "-generate_descriptors":
            generate_descs = True
        elif sys.argv[arg_index] == "-results_only":
            results_only = True
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

    db_sequences = sorted(os.listdir(results_db_path))

    file_output = os.path.join(results_db_path, desc_name + "_" + database[0:3] + "_basetest.csv")

    if not results_only:
        if os.path.exists(file_output):
            os.remove(file_output)

        for seq in db_sequences:
            if os.path.isdir(os.path.join(results_db_path, seq)):
                generate_results(desc_name, seq)

    if os.path.exists(file_output):
        seq_statistics = {}
        with open(file_output) as f:
            line = f.readline()
            while line != "":
                line_split = line.split(",")
                sequence = line_split[0][4:7]
                if sequence not in seq_statistics:
                    seq_statistics[sequence] = [0, 0, 0, 0, 0, 0]
                seq_statistics[sequence][MATCH_RATIO] += float(line_split[7])
                seq_statistics[sequence][MATCHING_SCORE] += float(line_split[8])
                seq_statistics[sequence][PRECISION] += float(line_split[9])
                seq_statistics[sequence][RECALL] += float(line_split[10])
                seq_statistics[sequence][MATCH_TIME] += int(line_split[11])
                seq_statistics[sequence][NUM_QUERIES] += 1

                line = f.readline()

        if len(seq_statistics) > 0:
            total_match_ratio = 0
            total_matching_score = 0
            total_precision = 0
            total_recall = 0
            total_match_time = 0
            total_num_queries = 0

            for seq in seq_statistics:
                print("Sequence " + seq +":")
                print("  Match ratio:", seq_statistics[seq][MATCH_RATIO] / seq_statistics[seq][NUM_QUERIES])
                print("  Matching score:", seq_statistics[seq][MATCHING_SCORE] / seq_statistics[seq][NUM_QUERIES])
                print("  Precision:", seq_statistics[seq][PRECISION] / seq_statistics[seq][NUM_QUERIES])
                print("  Recall:", seq_statistics[seq][RECALL] / seq_statistics[seq][NUM_QUERIES])
                print("  Match time (ms):", seq_statistics[seq][MATCH_TIME] / seq_statistics[seq][NUM_QUERIES])
                print()

                total_match_ratio += seq_statistics[seq][MATCH_RATIO]
                total_matching_score += seq_statistics[seq][MATCHING_SCORE]
                total_precision += seq_statistics[seq][PRECISION]
                total_recall += seq_statistics[seq][RECALL]
                total_match_time += seq_statistics[seq][MATCH_TIME]
                total_num_queries += seq_statistics[seq][NUM_QUERIES]

            print("Cumulative results:")
            print("  Match ratio:", total_match_ratio / total_num_queries)
            print("  Matching score:", total_matching_score / total_num_queries)
            print("  Precision:", total_precision / total_num_queries)
            print("  Recall:", total_recall / total_num_queries)
            print("  Match time (ms):", total_match_time / total_num_queries)
            print()




    else:
        print("No queries found.")
