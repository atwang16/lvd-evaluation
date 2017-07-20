import subprocess
import sys
import os.path
import os
import matplotlib.pylab as plt
import matplotlib.gridspec as gridspec

ROOT_PATH = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
results_path = None
image_db_path = None
distthresh_executable = None
descriptor_parameters_path = None
PARAMETERS_FILE_PATH = os.path.join(os.getcwd(), "parameters.txt")
IMG_EXTENSIONS = [".jpg", ".png", ".ppm", ".pgm"]
distance = "L2"
append = 0
generate_descs = False
DESCRIPTOR_SUFFIX = "ds.csv"
KEYPOINT_SUFFIX = "kp.csv"


def is_image(fname):
    _, ext = os.path.splitext(fname)
    return ext in IMG_EXTENSIONS


def generate_descriptors(desc, db):
    global descriptor_executable, descriptors_parameter_file, image_db_path, results_path
    args = [descriptor_executable, descriptors_parameter_file, image_db_path, results_path]
    print(desc + ": " + "extracting descriptors from " + db)
    print("***")
    subprocess.run(args)


def generate_results(sequence):
    global distthresh_executable, image_db_path, results_desc_path, results_db_path
    global append
    img_num = 2
    image_seq_path = os.path.join(image_db_path, sequence)
    results_seq_path = os.path.join(results_db_path, sequence)
    expand_image_dir = os.listdir(image_seq_path)

    images = []
    for file in expand_image_dir:
        if is_image(file):
            images.append(file)
    sorted(images)

    img1_base_name, _ = os.path.splitext(images[0])
    desc1_path = os.path.join(results_seq_path, img1_base_name + "_" + DESCRIPTOR_SUFFIX)
    kp1_path = os.path.join(results_seq_path, img1_base_name + "_" + KEYPOINT_SUFFIX)

    while img_num <= len(images):
        current_img = images[img_num - 1]
        img_base_name, _ = os.path.splitext(current_img)
        current_ds = img_base_name + "_" + DESCRIPTOR_SUFFIX
        current_kp = img_base_name + "_" + KEYPOINT_SUFFIX
        current_hom = "H1to" + str(img_num) + "p"

        desc2_path = os.path.join(results_seq_path, current_ds)
        kp2_path = os.path.join(results_seq_path, current_kp)
        hom_path = os.path.join(image_seq_path, current_hom)

        args = [distthresh_executable, PARAMETERS_FILE_PATH, desc_name, desc1_path, kp1_path, desc2_path, kp2_path,
                hom_path, distance, str(append), results_desc_path + os.sep]

        print(desc_name + ": " + database + " - " + sequence + ": " + images[0] + " to " + current_img)
        subprocess.run(args)
        img_num += 1
        append = 1


# MAIN #

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python distancethresh.py desc_name database_name [-generate_descriptors] [-append]")
        sys.exit(1)

    desc_name = sys.argv[1]
    database = sys.argv[2]

    arg_index = 3
    while arg_index < len(sys.argv):
        if sys.argv[arg_index] == "-generate_descriptors":
            generate_descs = True
        if sys.argv[arg_index] == "-append":
            append = 1
        arg_index += 1

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
                image_db_path = os.path.join(ROOT_PATH, directory, database)
            elif var == "DISTTHRESH_EXECUTABLE":
                distthresh_executable = os.path.join(ROOT_PATH, directory)
            elif var == "DESCRIPTORS_FOLDER":
                descriptor_parameters_path = os.path.join(ROOT_PATH, directory)
            line = f.readline()

    if generate_descs:
        generate_descriptors(desc_name, database)

    results_desc_path = os.path.join(results_path, desc_name)
    results_db_path = os.path.join(results_path, desc_name, database)

    with open(os.path.join(descriptor_parameters_path, desc_name + "_parameters.txt")) as f:
        line = f.readline()
        while line != "":
            line_split = line.split("=")
            var = line_split[0]
            value = line_split[-1]
            if value[-1] == "\n":
                value = value[:-1]
            if var == "distance":
                distance = value
            line = f.readline()

    db_sequences = os.listdir(results_db_path)

    for seq in db_sequences:
        if os.path.isdir(os.path.join(results_db_path, seq)):  # ignore .DS_STORE-like files
            generate_results(seq)

    # Plot recorded distances
    gs = gridspec.GridSpec(3, 1)
    fig = plt.figure()

    with open(os.path.join(results_desc_path, desc_name + "_cor_dists.csv")) as f:
        allpos_subplot = fig.add_subplot(gs[0])
        corr_match_distances = []
        line = f.readline()
        while line != "" and line != "\n":
            corr_match_distances.append(float(line))
            line = f.readline()

        allpos_subplot.hist(corr_match_distances, bins=40, color='g')

    with open(os.path.join(results_desc_path, desc_name + "_pos_dists.csv")) as f:
        pos_subplot = fig.add_subplot(gs[1], sharex=allpos_subplot)
        good_match_distances = []
        line = f.readline()
        while line != "" and line != "\n":
            good_match_distances.append(float(line))
            line = f.readline()

        pos_subplot.hist(good_match_distances, bins=40, color='b')

    with open(os.path.join(results_desc_path, desc_name + "_neg_dists.csv")) as f:
        neg_subplot = fig.add_subplot(gs[2], sharex=allpos_subplot)
        bad_match_distances = []
        line = f.readline()
        while line != "" and line != "\n":
            bad_match_distances.append(float(line))
            line = f.readline()

        neg_subplot.hist(bad_match_distances, bins=40, color='r')

    plt.savefig(os.path.join(results_desc_path, desc_name + "_dist_graph.png"), bbox_inches='tight')
    plt.show()
