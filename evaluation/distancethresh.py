import subprocess
import sys
import os.path
import os
import matplotlib.pylab as plt
import matplotlib.gridspec as gridspec

ROOT_PATH = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
RESULTS_PATH = os.path.join(ROOT_PATH, "results")
IMG_DB_PATH = os.path.join(ROOT_PATH, "datasets")
DIST_THRESH_BUILD = os.path.join(ROOT_PATH, "build_distthresh", "distances")
PARAMETERS_FILE_PATH = os.path.join(ROOT_PATH, "evaluation", "parameters.txt")
DESCRIPTOR_PARAMETERS_DIR = os.path.join(ROOT_PATH, "descriptors")
IMG_EXTENSIONS = [".jpg", ".png", ".ppm", ".pgm"]
distance = "L2"
append = 0

def is_image(fname):
    _, ext = os.path.splitext(fname)
    return ext in IMG_EXTENSIONS

def generate_results(sf):
    img_num = 2
    image_sf_path = os.path.join(image_dir, sf)
    results_sf_path = os.path.join(ds_kp_path, sf)
    expand_image_dir = os.listdir(image_sf_path)
    global append

    images = []
    for file in expand_image_dir:
        if is_image(file):
            images.append(file)
    sorted(images)

    img1_base_name, _ = os.path.splitext(images[0])
    desc1_path = os.path.join(results_sf_path, img1_base_name + "_ds.csv")
    kp1_path = os.path.join(results_sf_path, img1_base_name + "_kp.csv")

    while img_num <= len(images):
        current_img = images[img_num - 1]
        img_base_name, _ = os.path.splitext(current_img)
        current_ds = img_base_name + "_ds.csv"
        current_kp = img_base_name + "_kp.csv"
        current_hom = "H1to" + str(img_num) + "p"

        desc2_path = os.path.join(results_sf_path, current_ds)
        kp2_path = os.path.join(results_sf_path, current_kp)
        hom_path = os.path.join(image_sf_path, current_hom)

        args = [DIST_THRESH_BUILD, PARAMETERS_FILE_PATH, desc_name, desc1_path, kp1_path, desc2_path, kp2_path,
                hom_path, distance, str(append), results_path + os.sep]

        print(desc_name + ": " + database + " - " + sf + ": " + images[0] + " to " + current_img)
        subprocess.run(args)
        img_num += 1
        append = 1

####### MAIN #######

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python distancethresh.py desc_name database_name [-append]")
        sys.exit(1)

    desc_name = sys.argv[1]
    database = sys.argv[2]
    if len(sys.argv) >= 4:
        append = 1 if sys.argv[3] == "-append" else 0
    image_dir = os.path.join(IMG_DB_PATH, database)
    results_path = os.path.join(RESULTS_PATH, desc_name)
    ds_kp_path = os.path.join(RESULTS_PATH, desc_name, database)

    with open(os.path.join(DESCRIPTOR_PARAMETERS_DIR, desc_name + "_parameters.txt")) as f:
        line = f.readline()
        while line != "":
            line_split = line.split("=")
            var = line_split[0]
            value = line_split[-1]
            if (value[-1] == "\n"):
                value = value[:-1]
            if var == "distance":
                distance = value
            line = f.readline()

    subfolders = os.listdir(ds_kp_path)

    for sf in subfolders:
        if os.path.isdir(os.path.join(ds_kp_path, sf)):  # ignore .DS_STORE-like files
            generate_results(sf)

    gs = gridspec.GridSpec(3, 1)
    fig = plt.figure()

    with open(os.path.join(results_path, desc_name + "_cor_dists.csv")) as f:
        allpos_subplot = fig.add_subplot(gs[0])
        corr_match_distances = []
        line = f.readline()
        while line != "" and line != "\n":
            corr_match_distances.append(float(line));
            line = f.readline()

        allpos_subplot.hist(corr_match_distances, bins=40, color='g')

    with open(os.path.join(results_path, desc_name + "_pos_dists.csv")) as f:
        pos_subplot = fig.add_subplot(gs[1], sharex = allpos_subplot)
        good_match_distances = []
        line = f.readline()
        while line != "" and line != "\n":
            good_match_distances.append(float(line));
            line = f.readline()

        pos_subplot.hist(good_match_distances, bins=40, color='b')

    with open(os.path.join(results_path, desc_name + "_neg_dists.csv")) as f:
        neg_subplot = fig.add_subplot(gs[2], sharex=allpos_subplot)
        bad_match_distances = []
        line = f.readline()
        while line != "" and line != "\n":
            bad_match_distances.append(float(line));
            line = f.readline()

        neg_subplot.hist(bad_match_distances, bins=40, color='r')

    plt.savefig(os.path.join(results_path, desc_name + "_dist_graph.png"), bbox_inches='tight')
    plt.show()
