import subprocess
import sys
import os.path
import os

MIN_NUM_ARGS = 3
ROOT_PATH = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
PROJECT_TREE = os.path.join(ROOT_PATH, "project_structure.txt")
directories = {"results_path": None, "datasets_path": None, "descriptors_path": None, "basetest_executable": None}
flags = {"-results": False}
results_ds_path = None
results_kp_path = None
descriptor_parameters_path = None
file_output = None
PARAMETERS_FILE_PATH = os.path.join(os.getcwd(), "parameters.txt")
IMG_EXTENSIONS = [".jpg", ".png", ".ppm", ".pgm"]
distance = "L2"
DESCRIPTOR_SUFFIX = "ds.csv"
KEYPOINT_SUFFIX = "kp.csv"
MATCH_RATIO = 0
MATCHING_SCORE = 1
PRECISION = 2
RECALL = 3
MATCH_TIME = 4
NUM_QUERIES = 5


def is_image(fname):
    _, ext = os.path.splitext(fname)
    return ext in IMG_EXTENSIONS


def generate_results(desc, sequence):
    global results_ds_path, results_kp_path, file_output
    img_num = 2
    image_seq_path = os.path.join(directories["datasets_path"], sequence)
    ds_seq_path = os.path.join(results_ds_path, sequence)
    kp_seq_path = os.path.join(results_kp_path, sequence)
    image_seq_expanded = sorted(os.listdir(image_seq_path))

    images = []
    for file in image_seq_expanded:
        if is_image(file):
            images.append(file)
    sorted(images)

    img1_path = os.path.join(image_seq_path, images[0])
    img1_base_name, _ = os.path.splitext(images[0])
    img_seq = images[0].split("_")[0] + "_" + images[0].split("_")[1]
    desc1_path = os.path.join(ds_seq_path, img1_base_name + "_" + DESCRIPTOR_SUFFIX)
    kp1_path = os.path.join(kp_seq_path, img1_base_name + "_" + KEYPOINT_SUFFIX)

    while img_num <= len(images):
        current_img = images[img_num - 1]
        img_base_name, _ = os.path.splitext(current_img)
        current_ds = img_base_name + "_" + DESCRIPTOR_SUFFIX
        current_kp = img_base_name + "_" + KEYPOINT_SUFFIX
        current_hom = "H1to" + str(img_num) + "p"

        img2_path = os.path.join(image_seq_path, current_img)
        desc2_path = os.path.join(ds_seq_path, current_ds)
        kp2_path = os.path.join(kp_seq_path, current_kp)
        hom_path = os.path.join(image_seq_path, current_hom)
        draw_results_file = os.path.join(ds_seq_path, img_seq + "_draw_001" + "_" + '{0:03d}'.format(img_num) + ".png")

        args = [directories["basetest_executable"], PARAMETERS_FILE_PATH, desc, img1_path, desc1_path, kp1_path, img2_path, desc2_path,
                kp2_path, hom_path, distance, "-s", file_output, "-d", draw_results_file]

        print(desc + ": " + database + " - " + sequence + ": " + images[0] + " to " + current_img)
        subprocess.run(args)
        img_num += 1

# MAIN #

if __name__ == "__main__":
    if len(sys.argv) < MIN_NUM_ARGS:
        print("Usage: python basetest.py desc_name database_name [-results_only]")
        sys.exit(1)

    desc_full_name = sys.argv[1]
    database = sys.argv[2]

    if desc_full_name.count("_") > 0:
        det_desc_split = desc_full_name.split("_")
        detector = det_desc_split[0]
        descriptor = det_desc_split[1]
    else:
        detector = descriptor = desc_full_name

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

    arg_index = MIN_NUM_ARGS
    while arg_index < len(sys.argv):
        for f in flags:
            if sys.argv[arg_index] == f:
                flags[f] = True
        arg_index += 1

    directories["datasets_path"] = os.path.join(directories["datasets_path"], database)
    results_ds_path = os.path.join(directories["results_path"], desc_full_name, database)
    results_kp_path = os.path.join(directories["results_path"], detector, database)
    descriptors_parameter_file = os.path.join(directories["descriptors_path"], descriptor + "_parameters.txt")

    with open(descriptors_parameter_file) as f:
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

    db_sequences = sorted(os.listdir(results_ds_path))

    file_output = os.path.join(results_ds_path, descriptor + "_" + database[0:3] + "_basetest.csv")

    if not flags["-results"]:
        if os.path.exists(file_output):
            os.remove(file_output)

        for seq in db_sequences:
            if os.path.isdir(os.path.join(results_ds_path, seq)):
                generate_results(descriptor, seq)

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
