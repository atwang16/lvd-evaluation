import subprocess
import sys
import os.path
import os
import matplotlib.pylab as plt
import matplotlib.gridspec as gridspec

root_folder = os.path.join(os.getcwd(), "..")
results_folder = "results"
image_folder = "datasets"
results_dir = os.path.join(root_folder, results_folder)
image_dir = os.path.join(root_folder, image_folder)
EXECUTABLE_PATH_DIR = os.path.join(root_folder, "distancethreshold", "distances")
PARAMETERS_FILE_PATH = os.path.join(root_folder, "evaluation", "parameters.txt")
DESCRIPTOR_PARAMETERS_DIR = os.path.join(root_folder, "descriptors")
distance = "L2"
append = "0"


def generate_results(path_addendum):
    img_num = 2
    image_sf_path = os.path.join(image_dir, path_addendum)
    results_sf_path = os.path.join(desc_kp_dir, path_addendum)
    expand_image_dir = os.listdir(image_sf_path)
    expand_descr_dir = os.listdir(results_sf_path)
    global append

    images = []
    homographies = []
    for file in expand_image_dir:
        if file[0] == "i":  # image; of form img#.ext
            images.append(file)
        elif file[0] == "H":  # homography; of form H1to#p
            homographies.append(file)
    sorted(images)
    sorted(homographies)

    descriptors = []
    keypoints = []
    for file in expand_descr_dir:
        if file[0] != "." and file[0:4] != "disp" and file[0:4] != "eval":  # not a .DS_STORE or result file
            if file[5] == "d":  # descriptor; of form img#_descriptor.csv
                descriptors.append(file)
            elif file[5] == "k":  # keypoint; of form img#_keypoint.csv
                keypoints.append(file)
    sorted(descriptors)
    sorted(keypoints)

    img1_path = os.path.join(image_sf_path, images[0])
    desc1_path = os.path.join(results_sf_path, descriptors[0])
    kp1_path = os.path.join(results_sf_path, keypoints[0])

    while img_num <= min([len(images), len(homographies), len(descriptors), len(keypoints)]):
        current_img = images[img_num - 1]
        current_desc = descriptors[img_num - 1]
        current_kp = keypoints[img_num - 1]

        img2_path = os.path.join(image_sf_path, current_img)
        desc2_path = os.path.join(results_sf_path, current_desc)
        kp2_path = os.path.join(results_sf_path, current_kp)
        hom_path = os.path.join(image_sf_path, homographies[img_num - 1])  # assuming there is an identity homography
        results_path = results_dir + os.sep

        args = [EXECUTABLE_PATH_DIR, PARAMETERS_FILE_PATH, desc_name, img1_path, desc1_path, kp1_path, img2_path,
                desc2_path,
                kp2_path, hom_path, distance, append, results_path]

        print(desc_name + ": " + database + " - " + path_addendum + ": " + images[0] + " to " + current_img)
        subprocess.run(args)
        img_num += 1
        append = "1"

####### MAIN #######

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python distancethresh.py desc_name database_name append")
        sys.exit(1)

    desc_name = sys.argv[1]
    database = sys.argv[2]
    append = "1" if sys.argv[3] == "1" else "0"
    image_dir = os.path.join(image_dir, database)
    results_dir = os.path.join(results_dir, desc_name)
    desc_kp_dir = os.path.join(results_dir, database)

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

    subfolders = os.listdir(desc_kp_dir)

    # for sf in subfolders:
    #     if sf[0] != "." and os.path.isdir(os.path.join(desc_kp_dir, sf)):  # ignore .DS_STORE-like files
    #         generate_results(sf)

    gs = gridspec.GridSpec(3, 1)
    fig = plt.figure()

    with open(os.path.join(results_dir, desc_name + "_allpos_dists.csv")) as f:
        allpos_subplot = fig.add_subplot(gs[0])
        corr_match_distances = []
        line = f.readline()
        while line != "" and line != "\n":
            corr_match_distances.append(float(line));
            line = f.readline()

        allpos_subplot.hist(corr_match_distances, bins=40, color='g')

    with open(os.path.join(results_dir, desc_name + "_pos_dists.csv")) as f:
        pos_subplot = fig.add_subplot(gs[1], sharex = allpos_subplot)
        good_match_distances = []
        line = f.readline()
        while line != "" and line != "\n":
            good_match_distances.append(float(line));
            line = f.readline()

        pos_subplot.hist(good_match_distances, bins=40, color='b')

    with open(os.path.join(results_dir, desc_name + "_neg_dists.csv")) as f:
        neg_subplot = fig.add_subplot(gs[2], sharex=allpos_subplot)
        bad_match_distances = []
        line = f.readline()
        while line != "" and line != "\n":
            bad_match_distances.append(float(line));
            line = f.readline()

        neg_subplot.hist(bad_match_distances, bins=40, color='r')

    plt.savefig(os.path.join(results_dir, desc_name + "_dist_graph.png"), bbox_inches='tight')
    plt.show()
